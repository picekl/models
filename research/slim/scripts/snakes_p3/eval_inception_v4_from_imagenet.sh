

#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=6


CROP_POSITION='0.2_0.2'
CROP_SIZE=0.6
MIRROR=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/evaluatios/inception_v4
CKPT=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4/step_checkpoints/model.ckpt-1000000

echo "$Evaluating $CKPT"
python eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=snakes2019p3 \
           --dataset_split_name=val \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=inception_v4 \
           --preprocessing_name=inception_test_crop \
           --modest=True \
           --moving_average_decay=0.999 \
           --crop_position="$CROP_POSITION" \
           --crop_size="$CROP_SIZE" \
           --save_fc=Predictions \
           --mirror=$MIRROR \
           --save_filenames=True \
           --save_fc=Predictions

