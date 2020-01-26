#!/bin/bash
##
# 2019, Lukas Picek ZCU | CMP
##

export CUDA_VISIBLE_DEVICES=7

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train2
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_new_slim_500_filtered_plus_val
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_new_slim_500_2

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=260000 \
    --save_interval_steps=20000 \
    --save_interval_secs=900 \
    --save_summaries_secs=900 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=16
