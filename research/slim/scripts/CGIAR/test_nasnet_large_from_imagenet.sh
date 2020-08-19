#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=6


#TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/SvampeAtlas/evaluatios/inception_resnet_v2_plantclef_500
#CKPT=/mnt/datagrid/personal/picekluk/SvampeAtlas/checkpoints/inception_resnet_v2_plantclef_500/step_checkpoints/model.ckpt-140000


TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/CGIAR/tfrecords
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/CGIAR/test/nasnet-a_large/
CKPT=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/nasnet-a_large/step_checkpoints/model.ckpt-10000


echo "$Evaluating $CKPT"
python3 eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=CGIAR \
           --dataset_split_name=test \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=nasnet_large \
           --moving_average_decay=0.999 \
           --eval_image_size=500 \
           --modest=True
