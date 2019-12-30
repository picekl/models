#!/bin/bash
##
# 2019, Lukas Picek ZCU | CMP
##

export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_CBAM_500
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_iNat2017/model.ckpt-1000000

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=250000 \
    --save_interval_steps=10000 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1800 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=12 \
    --attention_module=cbam_block
