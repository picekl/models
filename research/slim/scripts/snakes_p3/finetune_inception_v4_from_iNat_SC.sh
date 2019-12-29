#!/bin/bash
##
# 2019, Lukas Picek ZCU | CMP
##

export CUDA_VISIBLE_DEVICES=1

TF_DATASET_DIR=/home/picekl/Projects/Snakes/tf_records_train
TF_TRAIN_DIR=/home/picekl/Projects/Snakes/checkpoints/inception_v4_SC6
TF_CHECKPOINT_PATH=/home/picekl/Projects/pretrained_models/inception_v4_iNat2017/inception_v4_iNat_448.ckpt

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --preprocessing_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --save_interval_secs=120 \
    --save_summaries_secs=120 \
    --modest=True \
    --batch_size=24 \
    --learning_rate_decay_type=CLR \
    --moving_average_decay=0.999 \
    --optimizer=momentum \
    --momentum=0.95 \
    --min_momentum=0.85 \
    --weight_decay=0.00001 \
    --learning_rate=0.001 \
    --max_learning_rate=0.005 \
    --step_size=500 \
    --max_number_of_steps=10000