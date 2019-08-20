# DeepCNN
Try to set better baseline for famous CNN models. Build model with Keras and train with Tensorflow session.



## 0. Requirements

* python==3.6
* tensorflow-gpu==1.13.1
* numpy==1.16.4
* pathlib=1.0.1



## 1. Datasets

- **Cifar10** (Auto-download)
- **Cifar100** (Auto-download)



## 2. Model Zoo

| #    | Model         | config    | Cifar10 | Cifar100 |
| ---- | ------------- | --------- | ------- | -------- |
|      | VGG           |           |         |          |
|      | Resnet        |           |         |          |
| 001  | Preact-Resnet | 18 layers | 0.947   |          |
| 002  |               | 34 layers | 0.949   |          |
| 003  |               | 50 layers | 0.940   |          |
|      | DenseNet      |           |         |          |



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

* [x] Mixup
* [ ] Manifold Mixup

### 4.3 Others



## 5. Acknowledgement

* This repo references [Fenghz's PyTorch implementation](https://github.com/FengHZ/mixupfamily). It is better to compare two implementations for understanding some details.

