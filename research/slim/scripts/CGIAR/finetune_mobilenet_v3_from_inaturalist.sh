#!/bin/bash
##
# 2017, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##


export CUDA_VISIBLE_DEVICES=4

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/CGIAR/tfrecords_trainval/
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/mobilenet_v3_large_224_1.0_float/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/v3-large_224_1.0_float/pristine/model.ckpt-540000

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=CGIAR \
    --dataset_split_name=train \
    --model_name=mobilenet_v3_large \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=save/Assign_13, save/Assign_12 \
    --max_number_of_steps=10000 \
    --save_interval_steps=2000 \
    --save_summaries_secs=1800 \
    --modest=True \
    --batch_size=32
