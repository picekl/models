#!/bin/bash
##
# 2020, Lukas Picek
# Currently in USA, Page, Arizona
##


export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/CGIAR/tfrecords
TF_EVAL_DIR=/mnt/datagrid/personal/picekluk/CGIAR/test/inception_v4_from_PlantClef2018_500/

mkdir -p $TF_EVAL_DIR

for IT in $(seq 2000 +2000 10000); do

    CKPT=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/inception_v4_from_PlantClef2018_500/step_checkpoints/model.ckpt-$IT

    echo "$Evaluating $CKPT"
    python eval_image_classifier.py \
           --checkpoint_path="${CKPT}" \
           --eval_dir=${TF_EVAL_DIR} \
           --dataset_name=CGIAR \
           --dataset_split_name=test \
           --dataset_dir=${TF_DATASET_DIR} \
           --model_name=inception_v4 \
           --moving_average_decay=0.999 \
           --eval_image_size=500 \
           --modest=True

done

