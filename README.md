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
- **Omniglot** ([Download Link](https://github.com/brendenlake/omniglot)): 50 alphabets, 15~40 characters in each alphabet, 20 samples per character



## 2. Model Zoo

**Note:** All the results listed below are computed on test set and I choice the better one from the final checkpoint and the best (on validation set) checkpoint. Validation set is random chosen from training set. Final checkpoint will always be saved and provide `--save_best_ckpt` option for saving best checkpoint (details please check `./run_scripts/*.sh` script files).

### 2.1 Vanilla Models

| #    | Model         | config     | Cifar10 | Cifar100 |
| ---- | ------------- | ---------- | ------- | -------- |
|      | VGG           |            |         |          |
|      | Resnet        |            |         |          |
| 001  | Preact-Resnet | 18 layers  | 0.947   | 0.748    |
| 002  |               | 34 layers  | 0.949   | 0.759    |
| 003  |               | 50 layers  | 0.940   |          |
|      | DenseNet      | 121 layers |         |          |

### 2.2 Improved Models

| #    | Model         | config    | Improvement | Cifar10 | Cifar100 |
| ---- | ------------- | --------- | ----------- | ------- | -------- |
|      | VGG           |           |             |         |          |
|      | Resnet        |           |             |         |          |
| 004  | Preact-Resnet | 18 layers | mixup       | 0.953   |          |
| 005  |               | 34 layers | mixup       | 0.955   |          |
| 006  |               | 50 layers | mixup       | 0.950   |          |
|      | DenseNet      |           |             |         |          |



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
* [ ] Resnet
* [x] Preact-Resnet
* [ ] DenseNet

### 4.2 Data Augmentation

* [x] Traditional augmentation
* [ ] More augmentation
* [x] Mixup
* [ ] Manifold mixup

### 4.3 Others



## 5. Acknowledgement

* This repo references [FengHZ's PyTorch implementation](https://github.com/FengHZ/mixupfamily). It is better to compare two implementations for understanding some details.
* Some API implementation referenced from PyTorch source code. 
