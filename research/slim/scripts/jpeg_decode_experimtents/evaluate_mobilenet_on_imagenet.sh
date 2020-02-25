#!/bin/bash
##
# 2020, Lukas Picek, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##


export CUDA_VISIBLE_DEVICES=7

TF_DATASET_DIR=/mnt/datagrid/public_datasets/imagenet/imagenet_fullres/tfrecord
CKPT=/mnt/datagrid/personal/picekluk/pretrained_models/mobilenet_v1/mobilenet_v1_1.0_224.ckpt.index
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/jpeg_coding_experiments/mobilenet_v1_1.0_224

mkdir -p $TF_EVAL_DIR

# Run evaluation for each ckeckpoint

python eval_image_classifier.py \
          --checkpoint_path=${CKPT} \
          --eval_dir=${TF_EVAL_DIR} \
          --dataset_name=imagenet \
          --dataset_split_name=validation \
          --dataset_dir=${TF_DATASET_DIR} \
          --model_name=mobilenet_v1 \
          --modest=True \
          --save_accuracy=True
