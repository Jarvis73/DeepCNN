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

"""
Load Omniglot dataset. Original dataset is split into two parts: images_background and
images_evaluation. Following [1] and [2], we merge two parts and randomly reconstruct
train/test set with 1200/423 characters.

(TODO: maybe no need)
Then we further leave out 150 characters for validation to avoid overfitting.


References:
    [1] Santoro A, Bartunov S, Botvinick M, et al. Meta-learning with memory-augmented
        neural networks[C]//International conference on machine learning. 2016: 1842-1850.
    [2] Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot
        learning[C]//Advances in neural information processing systems. 2016: 3630-3638.

"""

import cv2
import json
import pickle
import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path


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
    data_dir = Path(dataset_base_path) / "datasets/omniglot"

    if train:
        obj_file = data_dir / ("train_split_resized.pkl" if resize else "train_split.pkl")
    else:
        obj_file = data_dir / ("test_split_resized.pkl" if resize else "test_split.pkl")

    if obj_file.exists():
        with obj_file.open("rb") as f:
            dataset_ = pickle.load(f)
        with (data_dir / "dict_i2n.json").open() as f:
            i2n_ = json.load(f)
        with (data_dir / "dict_n2i.json").open() as f:
            n2i_ = json.load(f)
        return dataset_, n2i_, i2n_

    all_classes = []
    for split_dir in [data_dir / "images_background", data_dir / "images_evaluation"]:
        for language in split_dir.glob("*"):
            for character in language.glob("*"):
                all_classes.append("/".join(character.parts[-3:]))
    # name to index
    n2i = {name: idx for idx, name in enumerate(all_classes)}
    i2n = {idx: name for idx, name in enumerate(all_classes)}

    image_path = []
    global_labels = []
    for cls in all_classes:
        img_files = list((data_dir / cls).glob("*.png"))
        image_path.extend(img_files)
        global_labels.extend([n2i[cls]] * len(img_files))

    # Random split train/test sets
    idx = np.arange(len(all_classes))
    np.random.shuffle(idx)
    test_cls = idx[1200:]

    train_img = []
    train_lab = []
    test_img = []
    test_lab = []
    with tqdm.tqdm(total=len(global_labels)) as pbar:
        for img_file, lab in zip(image_path, global_labels):
            image, label = read_resize(img_file, lab)
            if label in test_cls:
                test_img.append(image)
                test_lab.append(label)
            else:
                train_img.append(image)
                train_lab.append(label)
            pbar.update(1)

    train_dataset = {"images": np.stack(train_img, axis=0), "labels": np.asarray(train_lab, dtype=np.uint16)}
    test_dataset = {"images": np.stack(test_img, axis=0), "labels": np.asarray(test_lab, dtype=np.uint16)}

    if resize:
        train_split_file = data_dir / "train_split_resized.pkl"
        test_split_file = data_dir / "test_split_resized.pkl"
    else:
        train_split_file = data_dir / "train_split.pkl"
        test_split_file = data_dir / "test_split.pkl"

    with train_split_file.open("wb") as f:
        pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
    with test_split_file.open("wb") as f:
        pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
    with (data_dir / "dict_n2i.json").open("w") as f:
        json.dump(n2i, f)
    with (data_dir / "dict_i2n.json").open("w") as f:
        json.dump(i2n, f)

    return train_dataset if train else test_dataset, n2i, i2n


def train_gen(classes, classes_map, x, num_cases, num_ways, num_shots, aug_class, seed=None, pbar=False):
    rng = np.random.RandomState(seed)
    if pbar:
        iter_range = tqdm.tqdm(range(num_cases))
    else:
        iter_range = range(num_cases)
    for _ in iter_range:
        selected_classes = rng.choice(classes, size=num_ways, replace=False)
        target_class = rng.choice(selected_classes, size=1, replace=False)[0]
        k_list = rng.randint(0, 4, size=num_ways)
        k_dict = {selected_class: k_item for selected_class, k_item in zip(selected_classes, k_list)}
        episode_labels = list(range(num_ways))
        class_to_episode_labels = {selected_class: episode_label
                                   for selected_class, episode_label in zip(selected_classes, episode_labels)}
        support_set_images = []
        support_set_labels = []
        for class_entry in selected_classes:
            choose_sample_list = rng.choice(classes_map[class_entry], size=num_shots, replace=False)
            class_image_samples = []
            for sample in choose_sample_list:
                x_data = x[sample]
                if aug_class:
                    x_data = np.rot90(x_data, k=k_dict[class_entry])
                class_image_samples.append(x_data)
            support_set_images.append(class_image_samples)
            support_set_labels.append([class_to_episode_labels[class_entry]] * num_shots)
        support_set_images = np.array(support_set_images, dtype=np.float32)     # [ways, shots, h, w, c]
        support_set_labels = np.array(support_set_labels, dtype=np.int32)       # [ways, shots]
        target_sample = rng.choice(classes_map[target_class], size=1, replace=False)[0]
        target_set_image = x[target_sample]
        if aug_class:
            target_set_image = np.rot90(target_set_image, k=k_dict[target_class])
        target_set_label = class_to_episode_labels[target_class]
        yield support_set_images, support_set_labels, target_set_image, target_set_label


def preprocess(x):
    x = np.asarray(x, dtype=np.float32) / 255.
    # x = (x - x.mean()) / x.std()
    x = (x - 0.919438) / 0.261186
    return x


def omniglot_dataset(dataset_base_path,
                     train_flag=True,
                     batch_size=1,
                     train_val_split=True,
                     num_val_cls=150,
                     num_ways=5,
                     num_shots=5,
                     num_train_batches_per_epoch=1000,
                     num_val_batches=250,
                     num_test_batches=1000,
                     augment_classes=True,
                     train_seed=None,
                     val_seed=1234,
                     test_seed=5678,
                     prefetch_buffer_size=None):
    dataset_, n2i, i2n = load_data(dataset_base_path, train_flag)
    x_data = preprocess(dataset_["images"][..., None])
    y_data = dataset_["labels"].astype(np.int32)
    total_classes_map = {}
    all_classes = np.unique(y_data)
    for i_ in all_classes:
        total_classes_map[i_] = np.where(y_data == i_)[0]

    def create_dataset(gen, ways_, shots_):
        dataset__ = tf.data.Dataset.from_generator(
            gen,
            (tf.float32, tf.int32, tf.float32, tf.int32),
            (tf.TensorShape([ways_, shots_, 28, 28, 1]),
             tf.TensorShape([ways_, shots_]),
             tf.TensorShape([28, 28, 1]),
             tf.TensorShape([])))
        return dataset__.batch(batch_size).prefetch(prefetch_buffer_size)

    if train_flag:
        dataset_length = num_train_batches_per_epoch * batch_size

        if train_val_split:
            rng = np.random.RandomState(train_seed)
            rng.shuffle(all_classes)
            train_classes = all_classes[:-num_val_cls]
            val_classes = all_classes[-num_val_cls:]
            train_classes_map = {i_: total_classes_map[i_] for i_ in total_classes_map if i_ in train_classes}
            val_classes_map = {i_: total_classes_map[i_] for i_ in total_classes_map if i_ in val_classes}
            val_dataset_length = num_val_batches * batch_size
            dataset_train = create_dataset(
                lambda: train_gen(train_classes, train_classes_map, x_data, dataset_length,
                                  num_ways, num_shots, augment_classes, train_seed), num_ways, num_shots)
            dataset_val = create_dataset(
                lambda: train_gen(val_classes, val_classes_map, x_data, val_dataset_length,
                                  num_ways, num_shots, False, val_seed, pbar=True), num_ways, num_shots)
            return {"train": {"data": dataset_train, "size": dataset_length},
                    "val": {"data": dataset_val, "size": val_dataset_length},
                    "i": n2i, "n": i2n, "shape": (batch_size, num_ways, num_shots, 28, 28, 1)}
        else:
            dataset_train = create_dataset(
                lambda: train_gen(all_classes, total_classes_map, x_data, dataset_length,
                                  num_ways, num_shots, augment_classes, train_seed), num_ways, num_shots)
            return {"train": {"data": dataset_train, "size": dataset_length},
                    "i": n2i, "n": i2n, "shape": (batch_size, num_ways, num_shots, 28, 28, 1)}
    else:
        dataset_length = num_test_batches * batch_size
        dataset_test = create_dataset(
                lambda: train_gen(all_classes, total_classes_map, x_data, dataset_length,
                                  num_ways, num_shots, False, test_seed, pbar=True), num_ways, num_shots)
        return {"test": {"data": dataset_test, "size": x_data.shape[0]},
                "i": n2i, "n": i2n, "shape": (batch_size, num_ways, num_shots, 28, 28, 1)}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ways = 10
    shots = 5
    dataset = omniglot_dataset(str(Path(__file__).parent.parent), train_flag=True, train_val_split=True,
                               num_ways=ways, num_shots=shots, augment_classes=True, train_seed=None)
    ai = dataset["train"]["data"].make_initializable_iterator()
    an = ai.get_next()
    sess = tf.Session()
    sess.run(ai.initializer)
    a, b, c, d = sess.run(an)
    print(a.shape)
    fig, ax = plt.subplots(shots + 1, ways)
    for i in range(shots + 1):
        for j in range(ways):
            if i < shots:
                ax[i, j].imshow(a[0, j, i, ..., 0], cmap="gray")
            elif j == d:
                ax[i, j].imshow(c[0, ..., 0], cmap="gray")
            ax[i, j].axis("off")
    plt.show()
    pass
