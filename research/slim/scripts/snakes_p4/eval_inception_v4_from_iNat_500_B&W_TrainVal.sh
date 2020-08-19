

#!/bin/bash
##
# 2019 Lukas Picek, UWB
##


export CUDA_VISIBLE_DEVICES=3


CROP_POSITION='0.2_0.2'
CROP_SIZE=0.8
MIRROR=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/tf_records
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/evaluatios/inception_v4_500_from_p4_bells_and_whistles_TrainVal_tests
CKPT=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/checkpoints/inception_v4_500_from_p4_bells_and_whistles_TrainVal/step_checkpoints/BW_trainval-model.ckpt-420000

echo "$Evaluating $CKPT"
python3 eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=snakes2020p4 \
           --dataset_split_name=test \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=inception_v4 \
           --preprocessing_name=inception_v4 \
           --moving_average_decay=0.9999 \
           --eval_image_size=500 \
           --modest=True \
           --save_fc=Predictions \
