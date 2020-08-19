
TRAIN_DIR=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/checkpoints/mobilenet_v3_3_360/
DATASET_DIR=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/tf_records
GRAPH_NAME=mobilenet_v3_360_109422.pb

python3 export_inference_graph.py \
   --alsologtostderr \
   --model_name=mobilenet_v3_custom \
   --dataset_dir=${DATASET_DIR} \
   --is_training=False \
   --dataset_name=svampeatlas_p1 \
   --output_file=${GRAPH_NAME}
