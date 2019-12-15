#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_CBAM
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/inception_v4_iNat2017/inception_v4_iNat_448.ckpt

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
    --max_number_of_steps=1000000 \
    --save_interval_secs=3600 \
    --save_summaries_secs=3600 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=32 \
    --attention_module=cbam_block
