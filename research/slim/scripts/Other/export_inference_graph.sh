
TRAIN_DIR=/home/picekl/Projects/Snakes/checkpoints/inception_v4_iNat/
DATASET_DIR=/home/picekl/Projects/Snakes/tf_records_train
GRAPH_NAME=inception_v4_from_iNat_1M.pb

python3 export_inference_graph.py \
   --alsologtostderr \
   --model_name=inception_v4 \
   --dataset_dir=${DATASET_DIR} \
   --is_training=False \
   --dataset_name=snakes2019p3 \
   --output_file=${GRAPH_NAME}