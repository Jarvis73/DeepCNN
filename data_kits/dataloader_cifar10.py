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

import os
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file

from lib import transforms


def load_data(dataset_base_path, train=True):
    """Loads CIFAR10 dataset.

    Parameters
    ----------
    dataset_base_path: str
        Path to create dataset dir, a recommended choice is project root dir
    train: bool
        flag, return training set or test set

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True, cache_dir=dataset_base_path)

    if train:
        num_train_samples = 50000
        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')
        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
             y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
        # y_train = np.reshape(y_train, (len(y_train), 1))
        y_train = np.asarray(y_train)
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose((0, 2, 3, 1))
        return x_train, y_train
    else:
        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)
        # y_test = np.reshape(y_test, (len(y_test), 1))
        y_test = np.asarray(y_test)
        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose((0, 2, 3, 1))
        return x_test, y_test


def cifar10_dataset(dataset_base_path,
                    train_flag=True,
                    batch_size=1,
                    train_val_split=True,
                    num_val_per_cls=500,
                    num_workers=None,
                    prefetch_buffer_size=None,
                    progress_bar=True):
    x_data, y_data = load_data(dataset_base_path, train_flag)
    y_data = y_data.astype(np.int32)

    def train_gen(x, y):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        for i_ in indices:
            yield x[i_], y[i_]

    def val_gen(x, y):
        indices = np.arange(x.shape[0])
        if progress_bar:
            for i_ in tqdm.tqdm(indices):
                yield x[i_], y[i_]
        else:
            for i_ in indices:
                yield x[i_], y[i_]

    def create_dataset(gen, trans):
        dataset_ = tf.data.Dataset.from_generator(
            gen, (tf.float32, tf.int32), (tf.TensorShape([32, 32, 3]), tf.TensorShape([])))
        dataset_ = (dataset_.map(lambda x_, y_: (trans(x_), y_), num_parallel_calls=num_workers)
                    .batch(batch_size)
                    .prefetch(prefetch_buffer_size))
        return dataset_

    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.Standardize(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ], tensor=True)

        if train_val_split:
            train_sample = []
            val_sample = []
            for i in range(10):
                cls_ind = np.where(y_data == i)[0]
                random_permed_cls_ind = np.random.permutation(cls_ind).tolist()
                train_sample.extend(random_permed_cls_ind[num_val_per_cls:])
                val_sample.extend(random_permed_cls_ind[:num_val_per_cls])
            dataset_val = create_dataset(lambda: val_gen(x_data[val_sample], y_data[val_sample]), transform)
            dataset_train = create_dataset(lambda: train_gen(x_data[train_sample], y_data[train_sample]), transform)
            return {"train": {"data": dataset_train, "size": len(train_sample)},
                    "val": {"data": dataset_val, "size": len(val_sample)}}
        else:
            dataset_train = create_dataset(lambda: train_gen(x_data, y_data), transform)
            return {"train": {"data": dataset_train, "size": x_data.shape[0]}}
    else:
        transform = transforms.Compose([
            transforms.Standardize(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ], tensor=True)
        dataset_test = create_dataset(lambda: val_gen(x_data, y_data), transform)
        return {"test": {"data": dataset_test, "size": x_data.shape[0]}}
