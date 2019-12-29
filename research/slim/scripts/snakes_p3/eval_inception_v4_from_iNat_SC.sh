

#!/bin/bash
##
# 2018, Milan Sulc, CTU in Prague
##
# The script is expected to run on server "lcgpu".
# Contains only training, not building tfrecord data files.
##

# As 2018 validation set I use the 2017 test set with GT annotations


export CUDA_VISIBLE_DEVICES=0

TF_DATASET_DIR=/home/picekl/Projects/Snakes/tf_records_train
TF_EVAL_DIR=/home/picekl/Projects/Snakes/evaluatios/inception_v4_iNat_SC
CKPT=/home/picekl/Projects/Snakes/checkpoints/inception_v4_SC

python3 eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CKPT} \
    --dataset_dir=${TF_DATASET_DIR} \
    --eval_dir=${TF_EVAL_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=val \
    --model_name=inception_v4 \
    --batch_size=10 \
    --modest=True
