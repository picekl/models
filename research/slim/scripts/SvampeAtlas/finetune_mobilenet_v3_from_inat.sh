#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=7

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/tf_records
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/checkpoints/mobilenet_v3_bells_and_whistles_from_iNat/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/mobilenet_v3/v3-large_224_1.0_float/ema/model-540000

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=svampeatlas_p1 \
    --dataset_split_name=train \
    --model_name=mobilenet_v3_large \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --ignore_missing_vars=True \
    --checkpoint_exclude_scopes=MobilenetV3/Logits,MobilenetV3/Predictions,MobilenetV3/predics \
    --max_number_of_steps=500000 \
    --save_interval_steps=10000 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1800 \
    --moving_average_decay=0.999 \
    --modest=True \
    --label_smoothing=0.001 \
    --learning_rate=0.032 \
    --replicas_to_aggregate=4 \
    --batch_size=128
