#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=2

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SvampeAtlas/tf_records
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/SvampeAtlas/evaluatios/mobilenet_v3
CKPT=/mnt/datagrid/personal/picekluk/SvampeAtlas/checkpoints/mobilenet_v3_2/step_checkpoints/model.ckpt-80000

echo "$Evaluating $CKPT"
python3 eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=svampeatlas_p1 \
           --dataset_split_name=validation \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=mobilenet_v3_large \
           --moving_average_decay=0.999 \
           --modest=True

