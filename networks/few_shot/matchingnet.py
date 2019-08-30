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

"""TF-Keras implementation of Matching Networks.

Matching networks were originally proposed in:
[1] Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot
    learning[C]//Advances in neural information processing systems. 2016: 3630-3638.


References:
    https://github.com/AntreasAntoniou/MatchingNetworks
    https://github.com/markdtw/matching-networks

"""

from ..base import BaseNet
import tensorflow as tf
from tensorflow import keras as K

L = K.layers


class VanillaEmbedding(BaseNet):
    def __init__(self, filter_list, name=None, **kwargs):
        super(VanillaEmbedding, self).__init__(name)
        self.embedding = []
        for i, num_filters in enumerate(filter_list):
            self.embedding.append(L.Conv2D(num_filters, 3, padding="same",
                                           name=self.name + ("layer%d/conv" % (i + 1)), **kwargs))
            self.embedding.append(L.BatchNormalization(name=self.name + ("layer%d/norm" % (i + 1))))
            self.embedding.append(L.ReLU(name=self.name + ("layer%d/relu" % (i + 1))))
            self.embedding.append(L.MaxPool2D(name=self.name + ("layer%d/pool" % (i + 1))))   # [bs, 1, 1, 64]
        self.embedding.append(L.Flatten())         # [bs, 64]

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.embedding:
            x = layer(x)
        return x


class FceG(BaseNet):
    def __init__(self, num_unit, name=None, **kwargs):
        super(FceG, self).__init__(name)
        self.num_unit = num_unit
        with tf.variable_scope(self.name[:-1] if self.name else "FceG"):
            self.bi_lstm = L.Bidirectional(L.LSTM(num_unit, return_sequences=True, **kwargs), merge_mode="sum")
            self.add = L.Add()

    def __call__(self, inputs, *args, **kwargs):
        input_shape = inputs.get_shape().as_list()
        assert self.num_unit == input_shape[-1], "FceG require num_unit == feature length of inputs, " \
                                                 "got {} vs {}".format(self.num_unit, input_shape[-1])
        h_sum = self.bi_lstm(inputs)
        # Skip connection
        return self.add([h_sum, inputs])


class FceF(BaseNet):
    def __init__(self, num_unit, process_steps, name=None, **kwargs):
        super(FceF, self).__init__(name)
        self.num_unit = num_unit
        self.process_steps = process_steps
        self.lstm = L.LSTMCell(num_unit, **kwargs)
        self.linear = L.Dense(num_unit)
        self.lambda_ = L.Lambda(self.forward)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: a list of two Tensors
            inputs: query image embedding with shape [batch_size, num_unit]
            content: support set images embedding with shape [batch_size, time_steps, num_unit]
        Returns
        -------
        """
        inputs, content = inputs
        with tf.variable_scope(self.name):
            bs, time_steps, _ = content.get_shape().as_list()
            # [batch_size, time_steps, 1]
            init_a = tf.ones((bs, time_steps, 1), dtype=content.dtype) / tf.constant(time_steps, tf.float32)
            # [batch_size, num_unit]
            r_km1 = tf.reduce_sum(content * init_a, axis=1)
            # LSTMCell init state
            state = [tf.zeros((bs, self.num_unit), dtype=inputs.dtype)] * 2

            new_h = tf.concat([state[1], r_km1], axis=1)
            # [batch_size, num_unit]
            state[1] = self.linear(new_h)

            for i in range(self.process_steps):
                _, state = self.lstm(inputs, state)
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
                new_h = tf.concat([state[1], r_km1], axis=1)
                state[1] = self.linear(new_h)
        return state[1]

    def __call__(self, inputs, *args, **kwargs):
        return self.lambda_(inputs)


class MatchingNetwork(BaseNet):
    """
    Matching network implementation

    Parameters
    ----------
    embedding_filter_list
    num_classes
    process_steps
    fce
    eps
    name
    """
    def __init__(self, embedding_filter_list, num_classes, process_steps, fce=False, sup_shape=None, eps=1e-10, name=None, **kwargs):
        super(MatchingNetwork, self).__init__(name or "MatchingNetwork")
        self.fce = fce
        self.eps = eps
        self.num_classes = num_classes
        self.sup_shape = sup_shape

        self.embedding = VanillaEmbedding(embedding_filter_list, name=self.name + "Embedding", **kwargs)
        self.fce_g = FceG(embedding_filter_list[-1], name=self.name + "FceG")
        self.fce_f = FceF(embedding_filter_list[-1], process_steps, name=self.name + "FceF")
        self.lambda_ = L.Lambda(self.forward, name=self.name[:-1])

    def forward(self, inputs):
        sup_inputs, sup_labels, que_inputs = inputs
        with tf.name_scope("Unroll"):
            bs, ways, shots, h, w, c = self.sup_shape
            sup_inputs = tf.reshape(sup_inputs, (bs, ways * shots, h, w, c))
            sup_labels = tf.reshape(sup_labels, (bs, ways * shots))
            sup_input_list = tf.unstack(sup_inputs, axis=1)
        sup_embedded = [self.embedding(sup_input) for sup_input in sup_input_list]
        que_embedded = self.embedding(que_inputs)
        if self.fce:
            # Apply full context embedding
            stacked_sup_embedded = tf.stack(sup_embedded, axis=1)
            g_encoded = self.fce_g(stacked_sup_embedded)
            f_encoded = self.fce_f((que_embedded, g_encoded))
        else:
            g_encoded = tf.stack(sup_embedded, axis=1)  # [bs, time_steps, num_unit]
            f_encoded = que_embedded                    # [bs, num_unit]

        # Compute cosine distance
        with tf.variable_scope(self.name + "Distance"):
            f_encoded = tf.expand_dims(f_encoded, axis=1)
            dot_product = tf.reduce_sum(g_encoded * f_encoded, axis=-1)     # [bs, time_steps]
            sup_norm = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(g_encoded ** 2, axis=-1), self.eps, float("inf")))
            # que_norm = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(f_encoded ** 2, axis=-1), self.eps, float("inf")))
            cosine_similarity = dot_product * sup_norm  # * que_norm

        with tf.name_scope(self.name + "Output"):
            att_kernel = tf.nn.softmax(cosine_similarity)           # [bs, time_steps]
            one_hot = tf.one_hot(sup_labels, self.num_classes)      # [bs, time_steps, num_classes]
            logits = tf.reduce_sum(tf.expand_dims(att_kernel, axis=-1) * one_hot, axis=1)     # [bs, num_classes]
            logits = logits / shots     # For normalization
        return logits

    def __call__(self, inputs, *args, **kwargs):
        return self.lambda_(inputs)
