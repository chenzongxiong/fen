import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import utils
from play import PILayer, PlayCell


sess = tf.Session()


def generator(inputs, weight, width, units, activation=None):
    # inputs, outputs_ = utils.load(fname)
    cell = PlayCell(weight=weight, width=width, debug=True)
    state = tf.constant(0, dtype=tf.float32)

    layer = PILayer(units=units, cell=cell,
                    activation=activation,
                    debug=True)
    outputs = layer(inputs, state)
    if activation is None:
        fname = "./training-data/players/{}-{}-{}-{}-linear.csv".format(method, weight, width, units)
    else:
        fname = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)

    outputs = outputs.eval(session=sess)

    outputs = outputs.T
    utils.save(inputs, outputs, fname)


if __name__ == "__main__":
    methods = ["sin"]
    widths = [1, 2, 3, 4, 5, 5.5, 6.5, 7, 9, 10, 12, 20, 100]
    weights = [1.0, 2.0, 3.0, 4.5]

    units = 4

    for method in methods:
        for weight in weights:
            for width in widths:
                print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                inputs, outputs_ = utils.load(fname)
                generator(inputs, weight, width, units, None)


    for method in methods:
        for weight in weights:
            for width in widths:
                print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                inputs, outputs_ = utils.load(fname)
                generator(inputs, weight, width, units, "tanh")
