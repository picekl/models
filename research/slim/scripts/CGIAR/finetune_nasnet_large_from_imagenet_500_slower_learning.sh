#!/bin/bash
##
# 2017, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##


export CUDA_VISIBLE_DEVICES=6

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/CGIAR/tfrecords_trainval/
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/nasnet-a_large_slower_learning/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/sulcmila/tf_data/models/nasnet-a_large/model.ckpt

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=CGIAR \
    --dataset_split_name=train \
    --model_name=nasnet_large \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=aux_11/aux_logits,final_layer/FC \
    --ignore_missing_vars=True \
    --max_number_of_steps=100000 \
    --save_interval_steps=2000 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1800 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=8 \
    --num_epochs_per_decay=3.0
