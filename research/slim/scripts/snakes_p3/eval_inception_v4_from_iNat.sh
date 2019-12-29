

#!/bin/bash
##
# 2019 Lukas Picek, UWB
##


export CUDA_VISIBLE_DEVICES=0


CROP_POSITION='0.1_0.1'
CROP_SIZE=8.0
MIRROR=0

TF_DATASET_DIR=/home/picekl/Projects/Snakes//tf_records_train

TF_EVAL_DIR=/home/picekl/Projects/Snakes/evaluatios/inception_v4_iNat

CKPT=/home/picekl/Projects/Snakes/checkpoints/inception_v4_iNat/model.ckpt-1000000

echo "$Evaluating $CKPT"
python3 eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=snakes2019p3 \
           --dataset_split_name=val \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=inception_v4 \
           --moving_average_decay=0.999 \
           --preprocessing_name=inception_v4 \
           --modest=True
