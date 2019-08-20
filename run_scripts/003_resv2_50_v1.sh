#!/usr/bin/env bash

TASK=$1
GPU_ID=$2
shift 2

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

if [[ "$TASK" == "train" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode train \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar10 \
        --batch_size 256 \
        --net-name resnet_v2_50 \
        --total_epochs 240 \
        --log_step 50 \
        --weight_decay 0.0005 \
        --learning_policy period_step \
        --learning_rate 0.1 \
        --lr_decay_step 60 \
        --lr_decay_rate 0.2 \
        --save_best_ckpt \
        $@
elif [[ "$TASK" == "test" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode test \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar10 \
        --net-name resnet_v2_50 \
        $@
elif [[ "$TASK" == "infer" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode infer \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar10 \
        --net-name resnet_v2_50 \
        $@
fi
