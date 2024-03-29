# DeepCNN
Try to set better baseline for famous CNN models. Build model with Keras and train with Tensorflow session.



## 0. Requirements

* python==3.6
* tensorflow-gpu==1.13.1
* numpy==1.16.4
* pathlib==1.0.1
* opencv==3.4.3


## 1. Datasets

- **Cifar10** (Auto-download): 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
- **Cifar100** (Auto-download): 60000 32x32 color images in 100 classes with 600 images per class. There are 50000 training images and 10000 test images. 
- **Omniglot** ([Download Link](https://github.com/brendenlake/omniglot)): 50 alphabets, 15~40 characters in each alphabet, 20 samples per character. There are 1623 classes, 32460 images(1623 * 20) in total. 964 classes for training and 659 classes for evaluation.



## 2. Model Zoo

**Note:** All the results listed below are computed on test set and I choice the better one from the final checkpoint and the best (on validation set) checkpoint. Validation set is random chosen from training set. Final checkpoint will always be saved and provide `--save_best_ckpt` option for saving best checkpoint (details please check `./run_scripts/*.sh` script files).

### 2.1 Vanilla Models

| #    | Model         | config     | Cifar10 | Cifar100 |
| ---- | ------------- | ---------- | ------- | -------- |
|      | VGG           |            |         |          |
| 11   | Resnet        | 18 layers  | 0.944   | 0.735    |
| 12   |               | 101 layers |         |          |
| 01   | Preact-Resnet | 18 layers  | 0.947   | 0.748    |
| 02   |               | 34 layers  | 0.949   | 0.759    |
| 03   |               | 50 layers  | 0.944   | 0.771    |
| 21   | DenseNet      | 121 layers | 0.945   | 0.762    |
| 22   |               | 100 layers | 0.948   |          |

### 2.2 Improved Models

| #    | Model         | config    | Improvement | Cifar10 | Cifar100 |
| ---- | ------------- | --------- | ----------- | ------- | -------- |
|      | VGG           |           |             |         |          |
|      | Resnet        |           |             |         |          |
| 04   | Preact-Resnet | 18 layers | mixup       | 0.953   |          |
| 05   |               | 34 layers | mixup       | 0.955   |          |
| 06   |               | 50 layers | mixup       | 0.950   |          |
|      | DenseNet      |           |             |         |          |

### 2.3 Few-Shot Models

| #    | Model            | config  | Omniglot |
| ---- | ---------------- | ------- | -------- |
| 01   | Matching Network | 5-w-1-s | 0.967    |
|      |                  |         |          |
|      |                  |         |          |



## 3. Usage

### A. For help

```bash
python main.py --help
```

### B. Train/Test with `bash` file

```bash
chmod u+x ./run_scripts/001_resv2_18_v1.sh
# For training with gpu-0
./run_scripts/001_resv2_18_v1.sh train 0
# For testing with gpu-1
./run_scripts/001_resv2_18_v1.sh test 1
```

One can re-run the bash files in `./run_scripts` directory to reproduce results above.



## 4. TODO List

### 4.1 Networks

* [ ] VGG
* [x] Resnet
* [x] Preact-Resnet
* [ ] Wide-Resnet
* [ ] ResNeXt
* [ ] Inception
* [ ] GoogLenet
* [ ] Xception
* [x] DenseNet
* [ ] MobileNet
* [ ] ShuffleNet
* [ ] SENet

### 4.2 Data Augmentation

* [x] Traditional augmentation
* [ ] More augmentation
* [x] Mixup
* [ ] Manifold mixup

### 4.3 Few-Shot Learning

* [ ] Matching Networks
* [ ] Siamese Networks
* [ ] Prototypical Networks



## 5. Acknowledgement

* This repo references [FengHZ's PyTorch implementation](https://github.com/FengHZ/mixupfamily). It is better to compare two implementations for understanding some details.
* Some API implementation referenced from PyTorch source code. 
