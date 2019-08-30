#!/usr/bin/env bash

TASK=$1
GPU_ID=$2
shift 2

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BASE_NAME=$(basename $0)

if [[ "$TASK" == "train" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main_fewshot.py \
        --mode train \
        --tag fs_${BASE_NAME%".sh"} \
        --dataset omniglot \
        --batch_size 32 \
        --net_name matchingnetwork \
        --total_epochs 200 \
        --log_step 200 \
        --optimizer Adam \
        --learning_policy custom_step \
        --lr_decay_boundaries 100 \
        --lr_custom_values 0.001 0.0001 \
        --save_best_ckpt \
        --num_ways 5 --num_shots 1 \
        $@
elif [[ "$TASK" == "test" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main_fewshot.py \
        --mode test \
        --tag fs_${BASE_NAME%".sh"} \
        --dataset omniglot \
        --batch_size 32 \
        --net_name matchingnetwork \
        $@
elif [[ "$TASK" == "infer" ]]; then
    PYTHONPATH=${PROJECT_DIR} PYTHONNOUSERSITE=True CUDA_VISIBLE_DEVICES=${GPU_ID} KMP_WARNINGS=0 python main_fewshot.py \
        --mode infer \
        --tag fs_${BASE_NAME%".sh"} \
        --dataset omniglot \
        --batch_size 32 \
        --net_name matchingnetwork \
        $@
fi
