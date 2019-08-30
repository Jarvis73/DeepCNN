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


def construct_data_batch(wrapper, mode, batch_size, train_val_split=True):
    if mode == "train":
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
    else:
        test_iter = wrapper["test"]["data"].make_initializable_iterator()
        wrapper["parent_iter"] = test_iter
        wrapper["test"]["steps"] = wrapper["test"]["size"] // batch_size

    return wrapper


def small_dataset(dataset, mode, batch_size, num_workers, train_val_split=True):
    if mode == "train":
        if dataset.lower() == "cifar10":
            from . import dataloader_cifar10
            wrapper = dataloader_cifar10.cifar10_dataset(
                "./", train_flag=True, batch_size=batch_size, train_val_split=train_val_split,
                num_workers=num_workers)
            wrapper["num_classes"] = 10
            wrapper["first_downsample"] = False
        elif dataset.lower() == "cifar100":
            from . import dataloader_cifar100
            wrapper = dataloader_cifar100.cifar100_dataset(
                "./", train_flag=True, batch_size=batch_size, train_val_split=train_val_split,
                num_workers=num_workers)
            wrapper["num_classes"] = 100
            wrapper["first_downsample"] = False
        else:
            raise ValueError("Not supported dataset")
    elif mode == "test":
        if dataset.lower() == "cifar10":
            from . import dataloader_cifar10
            wrapper = dataloader_cifar10.cifar10_dataset(
                "./", train_flag=False, batch_size=batch_size, num_workers=num_workers)
            wrapper["num_classes"] = 10
            wrapper["first_downsample"] = False
        elif dataset.lower() == "cifar100":
            from . import dataloader_cifar100
            wrapper = dataloader_cifar100.cifar100_dataset(
                "./", train_flag=False, batch_size=batch_size, num_workers=num_workers)
            wrapper["num_classes"] = 100
            wrapper["first_downsample"] = False
        else:
            raise ValueError("Not supported dataset")
    else:
        raise NotImplementedError
    return construct_data_batch(wrapper, mode, batch_size, train_val_split)


def few_shot_dataset(dataset, mode, batch_size, num_ways, num_shots,
                     train_val_split=True,
                     num_train_batches_per_epoch=1000,
                     num_val_batches=250,
                     num_test_batches=1000,
                     augment_classes=True,
                     train_seed=None,
                     val_seed=1234,
                     test_seed=5678):
    if mode == "train":
        if dataset.lower() == "omniglot":
            from . import dataloader_omniglot
            wrapper = dataloader_omniglot.omniglot_dataset("./",
                                                           train_flag=True,
                                                           batch_size=batch_size,
                                                           train_val_split=True,
                                                           num_val_cls=150,
                                                           num_ways=num_ways,
                                                           num_shots=num_shots,
                                                           num_train_batches_per_epoch=num_train_batches_per_epoch,
                                                           num_val_batches=num_val_batches,
                                                           augment_classes=augment_classes,
                                                           train_seed=train_seed,
                                                           val_seed=val_seed)
            wrapper["name"] = dataset
            wrapper["num_classes"] = num_ways
        else:
            raise ValueError("Not supported dataset")
    elif mode == "test":
        if dataset == "omniglot":
            from . import dataloader_omniglot
            wrapper = dataloader_omniglot.omniglot_dataset("./",
                                                           train_flag=False,
                                                           batch_size=batch_size,
                                                           num_ways=num_ways,
                                                           num_shots=num_shots,
                                                           num_test_batches=num_test_batches,
                                                           augment_classes=False,
                                                           test_seed=test_seed)
            wrapper["name"] = dataset
            wrapper["num_classes"] = num_ways
        else:
            raise ValueError("Not supported dataset")
    else:
        raise NotImplementedError
    return construct_data_batch(wrapper, mode, batch_size, train_val_split)
