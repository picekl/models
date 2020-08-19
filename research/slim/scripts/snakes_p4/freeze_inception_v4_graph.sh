

python3 /mnt/datagrid/personal/picekluk/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=inception_v4_SnakeRecognitionP4.pb \
  --input_checkpoint=/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/checkpoints/inception_v4_500_from_p4_bells_and_whistles/step_checkpoints/model.ckpt-680000 \
  --input_binary=true \
  --output_graph=frozen_inception_v4_SnakeRecognitionP4.pb \
  --output_node_names=InceptionV4/Logits/Predictions \
  --restore_op_name=save/restore_all \
