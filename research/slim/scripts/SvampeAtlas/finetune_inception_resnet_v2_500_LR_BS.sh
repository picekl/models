#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/tf_records
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/checkpoints/inception_resnet_v2_plantclef_500_bells_and_whistles_LR_BS/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/classification/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=svampeatlas_p1 \
    --dataset_split_name=train \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits/Logits,InceptionResnetV2/AuxLogits/Conv2d_2a_5x5 \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=500000 \
    --save_interval_steps=20000 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1800 \
    --modest=True \
    --batch_size=16 \
    --moving_average_decay=0.9999 \
    --modest=True \
    --label_smoothing=0.001 \
    --learning_rate=0.001 \
    --replicas_to_aggregate=8 \


