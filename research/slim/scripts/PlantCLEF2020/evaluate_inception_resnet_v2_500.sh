

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

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/PlantCLEF2020/tfrecords
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/PlantCLEF2020/evaluatios/inception_resnet_v2_plantclef_500/
CKPT=/mnt/datagrid/personal/picekluk/PlantCLEF2020/checkpoints/inception_resnet_v2_plantclef_500/step_checkpoints/model.ckpt-500000

echo "$Evaluating $CKPT"
python eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=PlantCLEF2020 \
           --dataset_split_name=validation \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=inception_resnet_v2 \
           --modest=True \
           --eval_image_size=500 \
           --moving_average_decay=0.999

