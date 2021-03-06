
export CUDA_VISIBLE_DEVICES=1

TF_DATASET_DIR=/home/picekl/Projects/Snakes/tf_records_train
TF_TRAIN_DIR=/home/picekl/Projects/Snakes/checkpoints/inception_v4
TF_CHECKPOINT_PATH=/home/picekl/Projects/pretrained_models/inception_v4_iNat2017/inception_v4_iNat_448.ckpt

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --max_number_of_steps=1000000 \
    --save_interval_secs=3600 \
    --save_summaries_secs=3600 \
    --moving_average_decay=0.999 \
    --batch_size=24 \
    --attention_module=cbam_block
