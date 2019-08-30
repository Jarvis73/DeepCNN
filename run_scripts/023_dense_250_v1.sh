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
        --net_name densenet_250 \
        --total_epochs 300 \
        --log_step 100 \
        --weight_decay 0.0001 \
        --learning_policy custom_step \
        --lr_decay_boundaries 150 225 \
        --lr_custom_values 0.1 0.01 0.001 \
        --momentum_use_nesterov \
        --drop_rate 0.2 \
        --init_channel 48 \
        --growth_rate 24 \
        --save_best_ckpt \
        $@
elif [[ "$TASK" == "test" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode test \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar10 \
        --batch_size 256 \
        --net_name densenet_250 \
        --init_channel 48 \
        --growth_rate 24 \
        $@
elif [[ "$TASK" == "infer" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main.py \
        --mode infer \
        --tag ${BASE_NAME%".sh"} \
        --dataset cifar10 \
        --batch_size 256 \
        --net_name densenet_250 \
        --init_channel 48 \
        --growth_rate 24 \
        $@
fi
