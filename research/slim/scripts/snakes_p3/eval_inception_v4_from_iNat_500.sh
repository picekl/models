

#!/bin/bash
##
# 2019 Lukas Picek, UWB
##


export CUDA_VISIBLE_DEVICES=0


CROP_POSITION='0.2_0.2'
CROP_SIZE=0.8
MIRROR=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/evaluatios/inception_v4_new_slim_500_filtered_plus_val
CKPT=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_new_slim_500_2/step_checkpoints/model.ckpt-250000

echo "$Evaluating $CKPT"
python3 eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=snakes2019p3 \
           --dataset_split_name=val \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=inception_v4 \
           --preprocessing_name=inception_v4 \
           --moving_average_decay=0.999 \
           --eval_image_size=500 \
           --modest=True
