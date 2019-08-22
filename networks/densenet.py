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

"""TF-Keras implementation of the DenseNets.

Densely connected networks (DenseNets) were proposed in:
[1] Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional
    networks[C]//Proceedings of the IEEE conference on computer vision and pattern
    recognition. 2017: 4700-4708.

http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf

References:

"""

from . import base

from tensorflow import keras as K

L = K.layers


class DenseLayer(base.BaseNet):
    def __init__(self, growth_rate, use_bottleneck=True, drop_rate=0, bn_params=None, name=None, **kwargs):
        super(DenseLayer, self).__init__(name)
        bn_params = bn_params or {}

        self.layers = []
        if use_bottleneck:
            self.layers.append(L.BatchNormalization(name=self.name + "norm1", **bn_params))
            self.layers.append(L.ReLU(name=self.name + "relu1"))
            self.layers.append(L.Conv2D(growth_rate * 4, 1, name=self.name + "conv1", **kwargs))
        self.layers.append(L.BatchNormalization(name=self.name + "norm2", **bn_params))
        self.layers.append(L.ReLU(name=self.name + "relu2"))
        self.layers.append(L.Conv2D(growth_rate, 3, name=self.name + "conv2", **kwargs))
        if drop_rate > 0.:
            self.layers.append(L.Dropout(drop_rate, name=self.name + "dropout"))

        self.concat = L.Concatenate(name=self.name + "concat")

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        out = self.concat([inputs, x])
        return out


class TransitionDown(base.BaseNet):
    def __init__(self, output_channels, bn_params=None, name=None, **kwargs):
        super(TransitionDown, self).__init__(name)
        bn_params = bn_params or {}

        self.layers = []
        self.layers.append(L.BatchNormalization(name=self.name + "norm", **bn_params))
        self.layers.append(L.ReLU(name=self.name + "relu"))
        self.layers.append(L.Conv2D(output_channels, 1, name=self.name + "conv", **kwargs))
        self.layers.append(L.AveragePooling2D(padding="same", name=self.name + "avg_pool"))

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class DenseBlock(base.BaseNet):
    def __init__(self, num_layers, growth_rate,
                 use_bottleneck=True, drop_rate=0., bn_params=None, name=None, **kwargs):
        super(DenseBlock, self).__init__(name)

        self.layers = []
        for i in range(num_layers):
            self.layers.append(DenseLayer(growth_rate, use_bottleneck, drop_rate, bn_params,
                                          name=self.name + "layer%d" % (i + 1), **kwargs))

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNet(base.BaseNet):
    def __init__(self,
                 base_channels,
                 num_classes,
                 layers_per_block,
                 growth_rate,
                 first_downsample=True,
                 use_bottleneck=True,
                 compression=0.5,
                 drop_rate=0.,
                 weight_decay=0.,
                 use_head=True,
                 bn_params=None,
                 name=None,
                 **kwargs):
        super(DenseNet, self).__init__(name)
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.base_channels = base_channels
        self.layers_per_block = layers_per_block
        self.growth_rate = growth_rate
        self.compression = compression
        self.bc = ""
        if use_bottleneck:
            self.bc += "B"
        if compression < 1:
            self.bc += "C"

        conv_params = {"padding": "same", "use_bias": True,
                       "kernel_regularizer": K.regularizers.l2(weight_decay) if weight_decay > 0 else None}
        conv_params.update(kwargs)
        bn_params = bn_params or {"momentum": 0.9, "epsilon": 1e-5}

        self.modules = []
        # Before dense blocks
        if first_downsample:
            self.modules.append(L.Conv2D(base_channels, 7, 2, name=self.name + "conv", **conv_params))
            self.modules.append(L.MaxPool2D(3, 2, padding="same", name=self.name + "pool"))
        else:
            self.modules.append(L.Conv2D(base_channels, 3, name=self.name + "conv", **conv_params))

        # Dense blocks
        output_channels = base_channels
        conv_params["use_bias"] = False
        for i, num_layers in enumerate(layers_per_block):
            self.modules.append(DenseBlock(num_layers, growth_rate, use_bottleneck, drop_rate, bn_params,
                                           name=self.name + "block%d" % (i + 1), **conv_params))
            output_channels += growth_rate * num_layers
            if i != len(layers_per_block) - 1:
                self.modules.append(TransitionDown(int(output_channels * compression),
                                                   name=self.name + "Td%d" % (i + 1), **conv_params))

        # After dense blocks
        self.modules.append(L.BatchNormalization(name=self.name + "norm", **bn_params))
        self.modules.append(L.ReLU(name=self.name + "relu"))
        if use_head:
            self.modules.append(L.GlobalAveragePooling2D(name=self.name + "g_avg_pool"))
            self.modules.append(L.Dense(num_classes, name=self.name + "fc"))

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for module in self.modules:
            x = module(x)
        return x

    def __repr__(self):
        str_ = self.__class__.__name__ + "(base_channels={}, layers={}, growth_rate={}, [BC]={}".format(
            self.base_channels, self.layers_per_block, self.growth_rate, self.bc)
        if self.compression < 1:
            str_ += ", compression={})".format(self.compression)
        else:
            str_ += ")"
        return str_


def densenet(layers=121,
             num_classes=1000,
             first_downsample=True,
             base_channels=None,
             growth_rate=None,
             layers_per_block=None,
             use_bottleneck=True,
             compression=0.5,
             drop_rate=0.,
             weight_decay=0.,
             use_head=True,
             bn_params=None,
             name=None,
             **kwargs):
    """
    Interface for DenseNet class. There are some pre-defined configurations
    for convenience. You can directly use `layers` in [121, 169, 201, 161]
    for the DenseNet instances mentioned in original paper. Or set `layers`
    to $6n + 4$ for depth [40, 100, 190, 250, ...] in which only three
    DenseBlocks are stacked. For other configs, just create DenseNet instance
    from class.

    Parameters
    ----------
    layers: int
        DenseNet model depth. [121, 169, 201, 161, 6n + 4] are supported.
    num_classes: int
        Classes for classification.
    first_downsample: bool
        Used in first conv layer. If true, conv7x7stride2 + maxpool3x3 will be
        used, else just conv3x3. Typically set true for ImageNet and false for
        Cifar data sets.
    base_channels: int
        Output channels of the first conv layer, which has impact on model
        performance.
    growth_rate: int
        Growth rate of densenet model. Details please reference paper.
    layers_per_block: list, tuple
        How many layers are stacked for dense connection in each DenseBlock.
    use_bottleneck: bool
        Use bottleneck or not in each DenseLayer.
    compression: float
        Channel compression rate in TransitionDown module. Default 0.5
    drop_rate: float
        Dropout rate, used at the end of each DenseLayer.
    weight_decay: float
        Weight decay rate for regularization.
    use_head: bool
        Whether add GlobalAveragePool+Dense layers at the top of the model.
        Default True.
    bn_params: dict
        Batch normalization parameters.
    name: str
        Model name
    kwargs: dict
        Convolution layer parameters.

    Returns
    -------
    A DenseNet instance. You can use __call__ function for build model.
    """
    params = {"num_classes": num_classes,
              "first_downsample": first_downsample,
              "use_bottleneck": use_bottleneck,
              "compression": compression,
              "drop_rate": drop_rate,
              "weight_decay": weight_decay,
              "use_head": use_head,
              "bn_params": bn_params,
              "name": name or ("densenet_%d" % layers)}
    params.update(kwargs)

    _configs = {
        121: {"growth_rate": 32, "layers": [6, 12, 24, 16]},
        169: {"growth_rate": 32, "layers": [6, 12, 32, 32]},
        201: {"growth_rate": 32, "layers": [6, 12, 48, 32]},
        161: {"growth_rate": 48, "layers": [6, 12, 36, 24]},
    }

    if layers in _configs:
        return DenseNet(base_channels=base_channels or _configs[layers]["growth_rate"] * 2,
                        layers_per_block=layers_per_block or _configs[layers]["layers"],
                        growth_rate=growth_rate or _configs[layers]["growth_rate"],
                        **params)
    elif use_bottleneck and (layers - 4) % 6 == 0:
        return DenseNet(base_channels=base_channels or 16,
                        layers_per_block=layers_per_block or [(layers - 4) // 6] * 3,
                        growth_rate=growth_rate or 12,
                        **params)
    else:
        raise NotImplementedError("No pre-defined config for layers={}. "
                                  "Only [121, 169, 201, 161, 6n + 4] layers are pre-defined. "
                                  "Please specify arguments `growth_rate`, `layers_per_block`, "
                                  "`use_bottleneck` and `compression` for custom densenet instance."
                                  .format(layers))
