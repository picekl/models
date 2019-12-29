

export CUDA_VISIBLE_DEVICES=1

TF_DATASET_DIR=/home/picekl/Projects/Snakes/tf_records_train
TF_TRAIN_DIR=/home/picekl/Projects/Snakes/checkpoints/inception_v4_SC_Test3
TF_CHECKPOINT_PATH=/home/picekl/Projects/Snakes/checkpoints/inception_v4_SC_Test3

mkdir -p $TF_TRAIN_DIR

python3 train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=snakes2019p3 \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --save_interval_secs=300 \
    --save_summaries_secs=300 \
    --modest=True \
    --batch_size=24 \
    --learning_rate_decay_type CLR \
    --optimizer momentum \
    --momentum 0.95 \
    --min_momentum 0.85 \
    --weight_decay 0.00001 \
    --learning_rate 0.1 \
    --max_learning_rate 0.5 \
    --step_size 1000 \
    --max_number_of_steps 10000