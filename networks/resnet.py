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

"""TF-Keras implementation of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

References:


"""

from . import base

from tensorflow import keras as K

L = K.layers


class ResBlock(base.BaseNet):
    def __init__(self,
                 input_channels,
                 output_channels,
                 strides=1,
                 drop_rate=0,
                 use_bottleneck=True,
                 bn_params=None,
                 name=None,
                 **kwargs):
        """
        Notice that this implementation of BasicBlock and Bottleneck do downsampling
        in the `first` conv3x3 layer. It seems that downsampling in first conv1x1 or
        conv3x3 will get similar results.

        See discussion in https://github.com/pytorch/vision/issues/191 for details.

        """
        super(ResBlock, self).__init__(name)

        self.layers = []
        if not use_bottleneck:
            self.layers.append(L.Conv2D(output_channels, 3, strides, name=self.name + "conv1", **kwargs))
            self.layers.append(L.BatchNormalization(name=self.name + "norm1", **bn_params))
            self.layers.append(L.ReLU(name=self.name + "relu1"))
            if drop_rate > 0:
                self.layers.append(L.Dropout(drop_rate, name=self.name + "dropout"))
            self.layers.append(L.Conv2D(output_channels, 3, 1, name=self.name + "conv2", **kwargs))
            self.layers.append(L.BatchNormalization(name=self.name + "norm2", **bn_params))
        else:
            self.layers.append(L.Conv2D(output_channels, 1, 1, name=self.name + "conv1", **kwargs))
            self.layers.append(L.BatchNormalization(name=self.name + "norm1", **bn_params))
            self.layers.append(L.ReLU(name=self.name + "relu1"))
            self.layers.append(L.Conv2D(output_channels, 3, strides, name=self.name + "conv2", **kwargs))
            self.layers.append(L.BatchNormalization(name=self.name + "norm2", **bn_params))
            self.layers.append(L.ReLU(name=self.name + "relu2"))
            if drop_rate > 0:
                self.layers.append(L.Dropout(drop_rate, name=self.name + "dropout"))
            self.layers.append(L.Conv2D(output_channels * 4, 1, name=self.name + "conv3", **kwargs))
            self.layers.append(L.BatchNormalization(name=self.name + "norm3", **bn_params))

        self.shortcuts = []
        if strides != 1 or use_bottleneck or input_channels != output_channels:
            # TODO(Alter): Maybe use max_pool when stride != 1
            self.shortcuts.append(L.Conv2D(output_channels * (4 if use_bottleneck else 1), 1, strides,
                                           name=self.name + "convs", **kwargs))
            self.shortcuts.append(L.BatchNormalization(name=self.name + "norms", **bn_params))
        # Must use keras.layers.add, instead of tf.add
        self.add = L.Add(name=self.name + "add")
        self.relu = L.ReLU(name=self.name + "relu")

    def __call__(self, inputs, *args, **kwargs):
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)
        shortcut = inputs
        for layer in self.shortcuts:
            shortcut = layer(shortcut)
        return self.relu(self.add([residual, shortcut]))


class ResModule(base.BaseNet):
    def __init__(self,
                 input_channels,
                 output_channels,
                 num_blocks,
                 strides,
                 drop_rate=0,
                 use_bottleneck=True,
                 bn_params=None,
                 name=None,
                 **kwargs):
        super(ResModule, self).__init__(name)

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(ResBlock(input_channels, output_channels, 1 if i != 0 else strides,
                                        drop_rate, use_bottleneck, bn_params,
                                        name=self.name + ("block%d" % (i + 1)), **kwargs))
            input_channels = output_channels * (4 if use_bottleneck else 1)

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for block in self.blocks:
            x = block(x)

        return x


class ResNet(base.BaseNet):
    def __init__(self,
                 base_channels,
                 num_classes,
                 block_num,
                 first_downsample=True,
                 drop_rate=0,
                 weight_decay=0.,
                 use_bottleneck=True,
                 use_head=True,
                 bn_params=None,
                 name=None,
                 **kwargs):
        super(ResNet, self).__init__(name)
        conv_params = {"padding": "same",
                       "kernel_regularizer": K.regularizers.l2(weight_decay) if weight_decay > 0. else None}
        conv_params.update(kwargs)
        bn_params = bn_params or {"momentum": 0.9, "epsilon": 1e-5}

        self.modules = []
        if first_downsample:
            self.modules.append(L.Conv2D(base_channels, 7, 2, name=self.name + "conv", **conv_params))
            self.modules.append(L.BatchNormalization(name=self.name + "norm", **bn_params))
            self.modules.append(L.ReLU(name=self.name + "relu"))
            self.modules.append(L.MaxPool2D(3, 2, padding="same", name=self.name + "pool"))
        else:
            self.modules.append(L.Conv2D(base_channels, 3, name=self.name + "conv", **conv_params))
            self.modules.append(L.BatchNormalization(name=self.name + "norm", **bn_params))
            self.modules.append(L.ReLU(name=self.name + "relu"))

        input_channels = base_channels
        output_channels = input_channels
        for i, num_blocks in enumerate(block_num):
            self.modules.append(ResModule(input_channels, output_channels, num_blocks,
                                          1 + int(i != 0), drop_rate, use_bottleneck, bn_params,
                                          name=self.name + ("module%d" % (i + 1)), **conv_params))
            input_channels = output_channels * (4 if use_bottleneck else 1)
            output_channels = output_channels * 2

        if use_head:
            self.modules.append(L.GlobalAveragePooling2D(name=self.name + "avgpool"))
            self.modules.append(L.Dense(num_classes, name=self.name + "fc"))

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for module in self.modules:
            x = module(x)
        return x


def resnet(layers,
           base_channels=64,
           num_classes=1000,
           first_downsample=True,
           drop_rate=0,
           kernel_initializer="he_normal",
           weight_decay=0.,
           use_head=True,
           bn_params=None,
           name=None,
           **kwargs):
    params = {"base_channels": base_channels,
              "num_classes": num_classes,
              "first_downsample": first_downsample,
              "drop_rate": drop_rate,
              "kernel_initializer": kernel_initializer,
              "weight_decay": weight_decay,
              "use_head": use_head,
              "bn_params": bn_params,
              "name": name or ("resnet_v2_%d" % layers)}
    params.update(kwargs)

    _configs = {
        18: {"block_num": [2, 2, 2, 2], "use_bottleneck": False},
        34: {"block_num": [3, 4, 6, 3], "use_bottleneck": False},
        50: {"block_num": [3, 4, 6, 3], "use_bottleneck": True},
        101: {"block_num": [3, 4, 23, 3], "use_bottleneck": True},
        152: {"block_num": [3, 8, 36, 3], "use_bottleneck": True},
        200: {"block_num": [3, 24, 36, 3], "use_bottleneck": True},
    }

    if layers in _configs:
        return ResNet(block_num=_configs[layers]["block_num"],
                      use_bottleneck=_configs[layers]["use_bottleneck"],
                      **params)
    else:
        raise NotImplementedError
