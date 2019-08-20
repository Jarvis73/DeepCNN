# Copyright 2019 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================


import tensorflow as tf
from tensorflow import keras as K
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

L = K.layers


class VanillaEmbedding(object):
    def __init__(self, filter_list, name=None):
        self.name = name or "Embedding"
        self.embedding = K.Sequential()
        for i, num_filters in enumerate(filter_list):
            self.embedding.add(L.Conv2D(num_filters, 3, padding="same"))
            self.embedding.add(L.BatchNormalization())
            self.embedding.add(L.ReLU())
            self.embedding.add(L.MaxPool2D())
        self.embedding.add(L.Flatten())

    def __call__(self, inputs, *args, **kwargs):
        return self.embedding(inputs, *args, **kwargs)


class FceG(object):
    def __init__(self, num_unit, name=None, **kwargs):
        self.name = name or "fce_g"
        self.num_unit = num_unit
        self.bi_lstm = L.Bidirectional(L.LSTM(num_unit, return_sequences=True, **kwargs), merge_mode="sum")

    def __call__(self, inputs, *args, **kwargs):
        input_shape = inputs.get_shape().as_list()
        assert self.num_unit == input_shape[-1], "FceG require num_unit == feature length of inputs, " \
                                                 "got {} vs {}".format(self.num_unit, input_shape[-1])
        h_sum = self.bi_lstm(inputs)
        # Skip connection
        output = h_sum + inputs
        return output


class FceF(object):
    def __init__(self, num_unit, process_steps, name=None, **kwargs):
        self.name = name or "fce_f"
        self.num_unit = num_unit
        self.process_steps = process_steps
        self.lstm = L.LSTMCell(num_unit, **kwargs)
        self.reuse = False

    def __call__(self, inputs, content, *args, **kwargs):
        """

        Parameters
        ----------
        inputs: Tensor
            query image embedding with shape [batch_size, num_unit]
        content: Tensor
            support set images embedding with shape [batch_size, time_steps, num_unit]
        args
        kwargs
        Returns
        -------
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            bs, time_steps, _ = content.get_shape().as_list()
            # [batch_size, time_steps, 1]
            init_a = tf.ones((bs, time_steps, 1), dtype=content.dtype) / tf.constant(time_steps, tf.float32)
            # [batch_size, num_unit]
            r_km1 = tf.reduce_sum(content * init_a, axis=1)
            # LSTMCell init state
            state = [tf.zeros((bs, self.num_unit), dtype=inputs.dtype)] * 2

            new_h = [state[1], r_km1]
            # [batch_size, num_unit]
            state[1] = seq2seq.Linear(new_h, self.num_unit, build_bias=True)(new_h)

            for i in range(self.process_steps):
                _, state = self.lstm(inputs, state)
                self.reuse = True
                if i == self.process_steps - 1:
                    break
                # [batch_size, num_unit]
                h = state[1] + inputs
                # [batch_size, 1, num_unit]
                h = tf.expand_dims(h, axis=1)
                # [batch_size, time_steps, 1]
                inner_prod = tf.reduce_sum(h, content, axis=2, keepdims=True)
                # [batch_size, time_steps, 1]
                a = tf.nn.softmax(inner_prod, axis=1)
                r_km1 = tf.reduce_sum(content * a, axis=1)
                new_h = [state[1], r_km1]
                state[1] = seq2seq.Linear(new_h, self.num_unit, build_bias=True)(new_h)
        return state[1]


class MatchingNetwork(object):
    def __init__(self, embedding_filter_list, process_steps, name=None):
        self.name = name or "MatchingNetwork"
        self.embedding = VanillaEmbedding(embedding_filter_list)
        self.fce_g = FceG(embedding_filter_list[-1])
        self.fce_f = FceF(embedding_filter_list[-1], process_steps)

    def __call__(self, sup_inputs, sup_labels, que_inputs):
        sup_input_list = tf.unstack(sup_inputs, axis=1)
        sup_embeded = [self.embedding(sup_input) for sup_input in sup_input_list]

