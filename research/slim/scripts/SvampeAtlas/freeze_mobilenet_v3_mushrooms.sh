python3 ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=mobilenet_v3_2.pb \
  --input_checkpoint=/mnt/datagrid/personal/picekluk/FGVC2018-Mushrooms/evaluatios/mobilenet_v3_2_300/ \
  --input_binary=true \
  --output_graph=frozen_mobilenet_v3_2.pb \
  --output_node_names=MobilenetV3/Predictions/Softmax
