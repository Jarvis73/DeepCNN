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

nn = K.layers


def get_model():
    inputs = K.Input(shape=(28, 28, 1))                         # 28x28@6
    out = nn.Conv2D(6, 5, activation=tf.nn.relu, padding="same")(inputs)        # 28x28@6
    out = nn.MaxPool2D()(out)                                   # 14x14@6
    out = nn.Conv2D(16, 5, activation=tf.nn.relu)(out)          # 10x10@16
    out = nn.MaxPool2D()(out)                                   # 5x5@16
    out = nn.Conv2D(120, 5, activation=tf.nn.relu)(out)         # 1x1@160
    out = nn.Flatten()(out)                                     # 160
    out = nn.Dense(84, activation=tf.nn.relu)(out)              # 84
    out = nn.Dense(10, activation=tf.nn.softmax)(out)           # 10
    return inputs, out


if __name__ == "__main__":
    x, y = get_model()
    model = K.Model(inputs=x, outputs=y)
    print(model.losses)
