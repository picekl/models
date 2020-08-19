
TRAIN_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/checkpoints/inception_v4_500_from_p4_bells_and_whistles
DATASET_DIR=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/tf_records
GRAPH_NAME=inception_v4_SnakeRecognitionP4_not_training.pb

python3 export_inference_graph.py \
   --alsologtostderr \
   --model_name=inception_v4 \
   --dataset_dir=${DATASET_DIR} \
   --is_training=False \
   --image_size=500 \
   --dataset_name=snakes2020p4 \
   --output_file=${GRAPH_NAME}
