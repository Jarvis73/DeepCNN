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
        --dataset cifar100 \
        --batch_size 64 \
        --net_name resnet_101 \
        --total_epochs 200 \
        --log_step 200 \
        --weight_decay 0.0005 \
        --learning_policy custom_step \
        --lr_decay_boundaries 60 120 160 \
        --lr_custom_values 0.1 0.02 0.004 0.0008 \
        --save_best_ckpt \
        $@
elif [[ "$TASK" == "test" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode test \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar100 \
        --batch_size 64 \
        --net_name resnet_101 \
        $@
elif [[ "$TASK" == "infer" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode infer \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar100 \
        --batch_size 64 \
        --net_name resnet_101 \
        $@
fi
