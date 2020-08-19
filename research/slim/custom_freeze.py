from __future__ import print_function
import tensorflow as tf
from nets.inception_v4 import inception_v4, inception_v4_arg_scope
from tensorflow.python.framework import graph_util
import sys
slim = tf.contrib.slim


output_node_names = "InceptionV4/Logits/Predictions"
output_graph_name = "./frozen_inception_v4_custom.pb"


input_checkpoint = '/mnt/datagrid/personal/picekluk/SnakeRecognition/P4/checkpoints/inception_v4_500_from_p4_bells_and_whistles/step_checkpoints/model.ckpt-680000'

with tf.Session(graph=tf.Graph(),config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=False)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ['add_default_attributes',
                      'remove_nodes(op=Identity, op=CheckNumerics)',
                      'fold_batch_norms', 'fold_old_batch_norms',
                      'strip_unused_nodes', 'sort_by_execution_order']
        transformed_graph_def = TransformGraph(tf.get_default_graph().as_graph_def(),'Placeholder', output_node_names.split(","), transforms)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            transformed_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(","))  # The output node names are used to select the useful nodes

        with tf.gfile.GFile("optimised_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())