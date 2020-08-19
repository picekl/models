#!/bin/bash
##
# 2019, Lukas Picek ZCU | CMP
##

export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/tf_records
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/checkpoints/inception_v4_500_from_p4_bells_and_whistles_lower_LR_BS_TrainVal/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_new_slim_500_2/

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2020p4 \
    --dataset_split_name=tarin_val \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=1000000 \
    --save_interval_steps=20000 \
    --save_interval_secs=900 \
    --save_summaries_secs=900 \
    --modest=True \
    --batch_size=16 \
    --moving_average_decay=0.9999 \
    --label_smoothing=0.001 \
    --replicas_to_aggregate=16 \
    --learning_rate=0.001 \
