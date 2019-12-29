

python3 ~/Projects/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=inception_v4_from_iNat_1M_AD_598.pb \
  --input_checkpoint=/home/picekl/Projects/Snakes/checkpoints/inception_v4_iNat/model.ckpt-1000000 \
  --input_binary=true \
  --output_graph=frozen_inception_v4_from_iNat_1M_AD_598.pb \
  --output_node_names=InceptionV4/Logits/Predictions