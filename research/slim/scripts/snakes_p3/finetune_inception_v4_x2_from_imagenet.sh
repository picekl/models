#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=7

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_x2
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_x2

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=train \
    --model_name=inception_v4_x2 \
    --preprocessing_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --max_number_of_steps=1000000 \
    --save_interval_steps=10000 \
    --save_interval_secs=3600 \
    --save_summaries_secs=3600 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=32 \
    --train_image_size=598
