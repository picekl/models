python3 /mnt/datagrid/personal/picekluk/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=mobilenet_v3_360_109422.pb \
  --input_checkpoint=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/checkpoints/mobilenet_v3_3_360/model.ckpt-109422 \
  --input_binary=true \
  --output_graph=frozen_mobilenet_v3_360_109422.pb \
  --output_node_names=MobilenetV3/Predictions/Softmax


