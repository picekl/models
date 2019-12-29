from __future__ import print_function
import tensorflow as tf
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from tensorflow.python.framework import graph_util
import sys
slim = tf.contrib.slim

checkpoint_file = '/home/picekl/Projects/Snakes/checkpoints/inception_v4_iNat/model.ckpt-1000000'

with tf.Graph().as_default() as graph:

    images = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32, name = 'input')

    with slim.arg_scope(inception_v3_arg_scope()):
        logits, end_points = inception_v3(images, num_classes = 3, create_aux_logits = False, is_training = False)

    variables_to_restore = slim.get_variables_to_restore()

    MOVING_AVERAGE_DECAY = 0.9999
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()        #This line is commented if EMA is turned off

    saver = tf.train.Saver(variables_to_restore)

    #Setup graph def
    input_graph_def = graph.as_graph_def()
    output_node_names = "InceptionV4/Predictions/Reshape_1"
    output_graph_name = "./frozen_inception_v4_new_100_221_ema.pb"

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)

        #Exporting the graph
        print ("Exporting graph...")
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(","))

        with tf.gfile.GFile(output_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())
