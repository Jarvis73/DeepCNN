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

"""TF-Keras implementation of the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

References:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py

"""

from tensorflow import keras as K

L = K.layers


class ResBlock(object):
    def __init__(self,
                 input_channels,
                 output_channels,
                 strides=1,
                 drop_rate=0, 
                 kernel_initializer="he_normal",
                 weight_decay=0.,
                 use_bottleneck=True, 
                 name=None):
        if name:
            name = name + "/"
        else:
            name = ""
        self.name = name

        regu = K.regularizers.l2(weight_decay)
        bn_params = {"momentum": 0.9, "epsilon": 1e-5}
        conv_params = {"use_bias": False, "kernel_initializer": kernel_initializer,
                       "kernel_regularizer": regu}

        self.layers = []
        if not use_bottleneck:
            self.layers.append(L.BatchNormalization(name=name + "norm1", **bn_params))
            self.layers.append(L.ReLU(name=name + "relu1"))
            self.layers.append(L.Conv2D(output_channels, 3, strides=strides, padding="same",
                                        name=name + "conv1", **conv_params))
            if drop_rate > 0:
                self.layers.append(L.Dropout(drop_rate, name=name + "dropout"))
            self.layers.append(L.BatchNormalization(name=name + "norm2", **bn_params))
            self.layers.append(L.ReLU(name=name + "relu2"))
            self.layers.append(L.Conv2D(output_channels, 3, strides=1, padding="same",
                                        name=name + "conv2", **conv_params))
        else:
            self.layers.append(L.BatchNormalization(name=name + "norm1", **bn_params))
            self.layers.append(L.ReLU(name=name + "relu1"))
            self.layers.append(L.Conv2D(output_channels, 1, name=name + "conv1", **conv_params))
            self.layers.append(L.BatchNormalization(name=name + "norm2", **bn_params))
            self.layers.append(L.ReLU(name=name + "relu2"))
            self.layers.append(L.Conv2D(output_channels, 3, strides=strides, padding="same",
                                        name=name + "conv2", **conv_params))
            if drop_rate > 0:
                self.layers.append(L.Dropout(drop_rate, name=name + "dropout"))
            self.layers.append(L.BatchNormalization(name=name + "norm3", **bn_params))
            self.layers.append(L.ReLU(name=name + "relu3"))
            self.layers.append(L.Conv2D(output_channels * 4, 1, name=name + "conv3", **conv_params))
        
        self.shortcuts = []
        if strides != 1 or use_bottleneck or input_channels != output_channels:
            # TODO(Alter): Maybe use max_pool when stride != 1
            self.shortcuts.append(L.BatchNormalization(name=name + "norms", **bn_params))
            self.shortcuts.append(L.ReLU(name=name + "relus"))
            self.shortcuts.append(L.Conv2D(output_channels * (4 if use_bottleneck else 1), 1, strides=strides,
                                           name=name + "convs", **conv_params))

    def __call__(self, inputs):
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)
        shortcut = inputs
        for layer in self.shortcuts:
            shortcut = layer(shortcut)
        # Must use keras.layers.add, instead of tf.add
        return L.add([residual, shortcut], name=self.name + "add")


class ResModule(object):
    def __init__(self,
                 input_channels,
                 output_channels,
                 num_blocks, 
                 strides,
                 drop_rate=0, 
                 kernel_initializer="he_normal",
                 weight_decay=0.,
                 use_bottleneck=True, 
                 name=None):
        if name:
            name = name + "/"
        else:
            name = ""

        block_params = {"drop_rate": drop_rate, "kernel_initializer": kernel_initializer,
                        "weight_decay": weight_decay, "use_bottleneck": use_bottleneck}

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(ResBlock(input_channels, output_channels, 1 if i != 0 else strides,
                                        name=name + ("block%d" % (i + 1)), **block_params))
            input_channels = output_channels * (4 if use_bottleneck else 1)

    def __call__(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        
        return x


class ResNetV2(object):
    def __init__(self,
                 base_channels, 
                 num_classes,
                 block_num,
                 first_downsample=True,
                 drop_rate=0,
                 kernel_initializer="he_normal",
                 weight_decay=0.,
                 use_bottleneck=True, 
                 use_head=True,
                 name=None):
        if name:
            name = name + "/"
        else:
            name = ""

        regu = K.regularizers.l2(weight_decay)
        block_params = {"drop_rate": drop_rate, "kernel_initializer": kernel_initializer,
                        "weight_decay": weight_decay, "use_bottleneck": use_bottleneck}
        conv_params = {"padding": "same", "use_bias": True, "kernel_initializer": kernel_initializer,
                       "kernel_regularizer": regu}
        bn_params = {"momentum": 0.9, "epsilon": 1e-5}
        
        self.modules = []
        if first_downsample:
            self.modules.append(L.Conv2D(base_channels, 7, 2, name=name + "conv", **conv_params))
            self.modules.append(L.MaxPool2D(3, 2, padding="same", name=name + "pool"))
        else:
            self.modules.append(L.Conv2D(base_channels, 3, name=name + "conv", **conv_params))

        input_channels = base_channels
        output_channels = input_channels
        for i, num_blocks in enumerate(block_num):
            self.modules.append(ResModule(input_channels, output_channels, num_blocks,
                                          strides=1 + int(i != 0), **block_params,
                                          name=name + ("module%d" % (i + 1))))
            input_channels = output_channels * (4 if use_bottleneck else 1)
            output_channels = output_channels * 2

        # Final bn+relu
        self.modules.append(L.BatchNormalization(name=name + "postnorm", **bn_params))
        self.modules.append(L.ReLU(name=name + "postrelu"))
        if use_head:
            self.modules.append(L.GlobalAveragePooling2D(name=name + "avgpool"))
            self.modules.append(L.Dense(num_classes, name=name + "fc"))
            # self.modules.append(L.Softmax(name=name + "softmax"))

    def __call__(self, inputs):
        x = inputs
        for module in self.modules:
            x = module(x)
        return x


def resnet_v2(layers,
              base_channels=64,
              num_classes=1000,
              first_downsample=True,
              drop_rate=0,
              kernel_initializer="he_normal",
              weight_decay=0.,
              use_head=True,
              name=None):
    kwargs = {"base_channels": base_channels,
              "num_classes": num_classes,
              "first_downsample": first_downsample,
              "drop_rate": drop_rate,
              "kernel_initializer": kernel_initializer,
              "weight_decay": weight_decay,
              "use_head": use_head,
              "name": name or ("resnet_v2_%d" % layers)}
    if layers == 18:
        return ResNetV2(block_num=[2, 2, 2, 2], use_bottleneck=False, **kwargs)
    elif layers == 34:
        return ResNetV2(block_num=[3, 4, 6, 3], use_bottleneck=False, **kwargs)
    elif layers == 50:
        return ResNetV2(block_num=[3, 4, 6, 3], use_bottleneck=True, **kwargs)
    elif layers == 101:
        return ResNetV2(block_num=[3, 4, 23, 3], use_bottleneck=True, **kwargs)
    elif layers == 152:
        return ResNetV2(block_num=[3, 8, 36, 3], use_bottleneck=True, **kwargs)
    elif layers == 200:
        return ResNetV2(block_num=[3, 24, 36, 3], use_bottleneck=True, **kwargs)
    else:
        raise NotImplementedError
