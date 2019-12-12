

#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=4


CROP_POSITION='0.0_0.0'
CROP_SIZE=1.0
MIRROR=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/tf_records_train

TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/evaluatios/inception_v4_iNat_SC5

CKPT=/mnt/datagrid/personal/picekluk/SnakeRecognition/checkpoints/inception_v4_iNat2017_SC5/model.ckpt-11071

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
           --crop_position="$CROP_POSITION" \
           --crop_size="$CROP_SIZE" \
           --save_fc=Predictions \
           --mirror=$MIRROR \
           --save_filenames=True

