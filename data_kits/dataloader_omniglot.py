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

import cv2
import pickle
import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path

from lib import transforms


def read_resize(img_file, lab):
    # Resize all images to 28x28
    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (28, 28))
    return img, lab


def read(img_file, lab):
    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
    return img, lab


def load_data(dataset_base_path, train=True, resize=True):
    """Loads Omniglot dataset.
    """
    datadir = Path(dataset_base_path) / "datasets/omniglot"

    if train:
        objfile = datadir / ("images_background_resized.pkl" if resize else "images_background.pkl")
        basedir = datadir / "images_background"
    else:
        objfile = datadir / ("images_evaluation_resized.pkl" if resize else "images_evaluation.pkl")
        basedir = datadir / "images_evaluation"

    if objfile.exists():
        with objfile.open("rb") as f:
            dataset = pickle.load(f)
            return dataset

    all_classes = []
    for language in basedir.glob("*"):
        for character in language.glob("*"):
            all_classes.append("/".join(character.parts[-2:]))
    # name to index
    n2i = {name: idx for idx, name in enumerate(all_classes)}
    i2n = {idx: name for idx, name in enumerate(all_classes)}

    image_path = []
    labels = []
    for cls in all_classes:
        cls_dir = basedir / cls
        img_files = list(cls_dir.glob("*.png"))
        image_path.extend(img_files)
        labels.extend([n2i[cls]] * len(img_files))

    images_ = []
    labels_ = []
    with tqdm.tqdm(total=len(labels)) as pbar:
        for img_file, lab in zip(image_path, labels):
            image, label = read_resize(img_file, lab)
            pbar.update(1)
            images_.append(image)
            labels_.append(label)

    dataset = {"images": np.stack(images_, axis=0),
               "labels": np.asarray(labels_, dtype=np.uint16),
               "n2i": n2i,
               "i2n": i2n}
    with objfile.open("wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    return dataset


def omniglot_dataset(dataset_base_path,
                     train_flag=True,
                     batch_size=1,
                     train_val_split=True,
                     num_val_per_cls=500,
                     num_workers=None,
                     prefetch_buffer_size=None,
                     progress_bar=True):
    # TODO: not finished
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

