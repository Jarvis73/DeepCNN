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

import argparse
from .resnet_v2 import resnet_v2
from .densenet import densenet

__all__ = [
    "resnet_v2",
    "densenet"
]


def checker(net):
    if not isinstance(net, str):
        raise argparse.ArgumentTypeError("`net_name` must be a string")
    parts = net.split("_")

    net_name = parts[0]
    if parts[1] == "v2":
        net_name += "_v2"
    if net_name not in __all__:
        raise argparse.ArgumentError("`net-name` must start with {}, but got {}"
                                     .format(__all__, net_name))
    return net
