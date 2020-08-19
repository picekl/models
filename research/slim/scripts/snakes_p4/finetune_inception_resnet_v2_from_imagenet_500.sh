#!/bin/bash
##
# 2019, Lukas Picek ZCU | CMP
##

export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/tf_records
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/checkpoints/inception_resnet_v2_500/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/classification/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2020p4 \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=1000000 \
    --save_interval_steps=20000 \
    --save_interval_secs=900 \
    --save_summaries_secs=900 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=16
