from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2 as cv
import numpy as np
import argparse
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=224, input_width=224,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    file_name = "tf_files/tick_photos/deer_tick/3.jpg"
    model_file = "tf_files/tick_graph.pb"
    label_file = "tf_files/tick_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        end = time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    # ---------SHOW----IMAGES---------

    img = cv.imread(file_name, cv.IMREAD_UNCHANGED)
    resized = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)
    result = resized.copy()
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    gray3 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    edges = cv.Canny(gray3, 100, 200)

    cv.putText(img, "1) Original", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv.putText(resized, "2) Resized", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv.putText(gray3, "3) Grayscaled", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv.putText(edges, "3) Edge Detection", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv.putText(result, "{} {:.4f}%".format(labels[top_k[0]], results[top_k[0]]), (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv.imshow('original', img)
    cv.imshow('resized', resized)
    cv.imshow('grayscaled', gray3)
    cv.imshow('edges', edges)
    cv.imshow('result', result)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    for i in top_k:
        print(template.format(labels[i], results[i]))

    # ---------EXIT----LOOP---------

    print("PRESS <ESC> TO EXIT.")

    while True:
        k = cv.waitKey(33)
        if k == 27:
            break

    cv.destroyAllWindows()
