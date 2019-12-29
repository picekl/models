import sys

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


def predict(img_path):

    graph = load_graph("frozen_inception_v4_test.pb")
    INPUT_NODE = graph.get_tensor_by_name('input:0')
    OUTPUT_NODE = graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')

    SESSION = tf.Session(graph=graph, config=config)

    im = Image.open(img_path)
    img = np.array(im)
    img = cv2.resize(img, dsize=(299, 299))
    img = np.expand_dims(img, axis=0)
    predictions = SESSION.run(OUTPUT_NODE, feed_dict={INPUT_NODE: img})

    return predictions

