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
from . import dataloader_cifar10, dataloader_cifar100


def small_dataset(dataset, mode, batch_size, num_workers, train_val_split=True):
    if mode == "train":
        if dataset == "cifar10":
            wrapper = dataloader_cifar10.cifar10_dataset(
                "./", train_flag=True, batch_size=batch_size, train_val_split=train_val_split,
                num_workers=num_workers)
            wrapper["num_classes"] = 10
            wrapper["first_downsample"] = False
        elif dataset == "cifar100":
            wrapper = dataloader_cifar100.cifar100_dataset(
                "./", train_flag=True, batch_size=batch_size, train_val_split=train_val_split,
                num_workers=num_workers)
            wrapper["num_classes"] = 100
            wrapper["first_downsample"] = False
        else:
            raise ValueError("Not supported dataset")

        train_iter = wrapper["train"]["data"].make_initializable_iterator()
        wrapper["train"]["iter"] = train_iter
        wrapper["train"]["steps"] = wrapper["train"]["size"] // batch_size
        wrapper["parent_iter"] = train_iter

        if train_val_split:
            val_iter = wrapper["val"]["data"].make_initializable_iterator()
            handler = tf.placeholder(tf.string, shape=(), name="Handler")
            iterator = tf.data.Iterator.from_string_handle(
                handler, train_iter.output_types, train_iter.output_shapes, train_iter.output_classes)
            wrapper["val"]["iter"] = val_iter
            wrapper["parent_iter"] = iterator
            wrapper["handler"] = handler
    elif mode == "test":
        if dataset == "cifar10":
            wrapper = dataloader_cifar10.cifar10_dataset(
                "./", train_flag=False, batch_size=batch_size, num_workers=num_workers)
            wrapper["num_classes"] = 10
            wrapper["first_downsample"] = False
        elif dataset == "cifar100":
            wrapper = dataloader_cifar100.cifar100_dataset(
                "./", train_flag=False, batch_size=batch_size, num_workers=num_workers)
            wrapper["num_classes"] = 100
            wrapper["first_downsample"] = False
        else:
            raise ValueError("Not supported dataset")
        test_iter = wrapper["test"]["data"].make_initializable_iterator()
        wrapper["parent_iter"] = test_iter
        wrapper["test"]["steps"] = wrapper["test"]["size"] // batch_size
    else:
        raise NotImplementedError
    return wrapper
