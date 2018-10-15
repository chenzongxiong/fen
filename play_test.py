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


def fit(inputs, outputs, units, activation, width, true_weight, method="sin"):

    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)

    cell = PlayCell(weight=weight, width=width, debug=True)
    # if activation == "linear":
    #     layer = PILayer(units=units, cell=cell,
    #                     activation=None,
    #                     debug=False)
    # else:
    layer = PILayer(units=units, cell=cell,
                    activation="tanh",
                    debug=False)
    predictions = layer(inputs, state)

    loss = tf.losses.mean_squared_error(labels=outputs,
                                        predictions=predictions)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    epochs = 1000
    loss_value = -1;
    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        print("epoch", i, "loss:", loss_value,
              ", true_weight: ", true_weight,
              ", activation: ", activation,
              ", weight: ", cell.kernel.eval(session=sess).tolist(),
              ", theta: ", layer.kernel.eval(session=sess).tolist(),
              ", bias: ", layer.bias.eval(session=sess).tolist())

    with open("./training-data/players/trained-{}-{}-{}-{}.txt".format(method, true_weight, width, activation), "w") as f:
        print("epochs: {}, loss: {}".format(epochs, loss_value))
        print("true weight: {}, estimated_weight: {}".format(true_weight, cell.kernel.eval(session=sess).tolist()))
        print("true theta: {}, estimated_theta: {}".format([[1], [2], [3], [4]], layer.kernel.eval(session=sess).tolist()))
        print("true bias: {}, estimated_bias: {}".format([[1], [2], [-1], [-2]], layer.bias.eval(session=sess).tolist()))

        f.write("epochs: {}, loss: {}\n".format(epochs, loss_value))
        f.write("true_weight: {}, estimated_weight: {}\n".format(true_weight, cell.kernel.eval(session=sess).tolist()))
        f.write("true theta: {}, estimated_theta: {}\n".format([[1], [2], [3], [4]], layer.kernel.eval(session=sess).tolist()))
        f.write("true bias: {}, estimated_bias: {}\n".format([[1], [2], [-1], [-2]], layer.bias.eval(session=sess).tolist()))


    print("================================================================================")



if __name__ == "__main__":
    methods = ["sin"]
    widths = [1, 2, 3, 4, 5, 5.5, 6.5, 7, 9, 10, 12, 20, 100]
    weights = [1.0, 2.0, 3.0, 4.5]

    units = 4

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #             inputs, outputs_ = utils.load(fname)
    #             generator(inputs, weight, width, units, None)


    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #             inputs, outputs_ = utils.load(fname)
    #             generator(inputs, weight, width, units, "tanh")

    activation = "tanh"
    for method in methods:
        for weight in weights:
            for width in widths:
                print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)
                inputs, outputs_ = utils.load(fname)
                fit(inputs, outputs_, units, activation, width, weight)


    activation = "linear"
    for method in methods:
        for weight in weights:
            for width in widths:
                print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)
                inputs, outputs_ = utils.load(fname)
                # generator(inputs, weight, width, units, None)
                fit(inputs, outputs_, units, activation, width, weight)
