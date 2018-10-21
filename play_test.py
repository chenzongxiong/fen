import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import utils
from play import Play, PlayCell


sess = tf.Session()


def generator(inputs, weight, width, units, activation=None, nbr_of_chunks=1):
    cell = PlayCell(weight=weight, width=width, debug=True)
    state = tf.constant(0, dtype=tf.float32)

    layer = Play(units=units, cell=cell,
                 activation=activation,
                 nbr_of_chunks=nbr_of_chunks,
                 debug=True)
    outputs = layer(inputs, state)

    init = tf.global_variables_initializer()
    sess.run(init)

    outputs = outputs.eval(session=sess)

    outputs = outputs.T
    return inputs, outputs


def fit(inputs, outputs, units, activation, width, true_weight, method="sin", nbr_of_chunks=1):

    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)

    cell = PlayCell(weight=weight, width=width, debug=False)

    layer = Play(units=units, cell=cell,
                 activation="tanh",
                 nbr_of_chunks=nbr_of_chunks,
                 debug=False)
    # predictions = layer(inputs, state)
    predictions = layer(inputs)

    loss = tf.losses.mean_squared_error(labels=outputs,
                                        predictions=predictions)
    loss_summary = tf.summary.scalar("loss", loss)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    utils.writer.add_graph(tf.get_default_graph())
    sess.run(init)

    epochs = 500
    loss_value = -1;
    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        if i % 10 == 0:
            summary = sess.run(loss_summary)
            utils.writer.add_summary(summary, i)

        print("epoch", i, "loss:", loss_value,
              ", true_weight: ", true_weight,
              ", activation: ", activation,
              ", weight: ", cell.kernel.eval(session=sess).tolist(),
              ", theta1: ", layer.kernel1.eval(session=sess).tolist(),
              ", theta2: ", layer.kernel2.eval(session=sess).tolist(),
              ", bias1: ", layer.bias1.eval(session=sess).tolist(),
              ", bias2: ", layer.bias2.eval(session=sess).tolist(),
              ", state:", layer.state.eval(session=sess).tolist())

    state = tf.constant(0, dtype=tf.float32)
    predictions = layer(inputs, state)

    return sess.run(predictions)



if __name__ == "__main__":
    utils.writer = utils.get_tf_summary_writer("./log/players/")

    methods = ["sin"]
    widths = [5]
    weights = [2]
    units = 4
    nbr_of_chunks = 1
    activation = "tanh"

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #             inputs, outputs_ = utils.load(fname)
    #             inputs, outputs = generator(inputs, weight, width, units, activation, nbr_of_chunks)
    #             if activation is None:
    #                 fname = "./training-data/players/{}-{}-{}-{}-{}-linear.csv".format(method, weight, width, units, nbr_of_chunks)
    #             else:
    #                 fname = "./training-data/players/{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, nbr_of_chunks, activation)
    #             utils.save(inputs, outputs, fname)


    # activation = "tanh"
    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)
    #             inputs, outputs = utils.load(fname)
    #             fname = "./pics/players/{}-{}-{}-{}-{}.pdf".format(method, weight, width, units, activation)
    #             plt.scatter(inputs, outputs)
    #             plt.savefig(fname)
    #             # utils.save_animation(inputs, outputs, fname)


    # activation = "tanh"
    _units = 4
    _nbr_of_chunks = 1

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = "./training-data/players/{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, nbr_of_chunks, activation)
    #             inputs, outputs_ = utils.load(fname)
    #             # increase *units* in order to increase the capacity of the model
    #             predictions = fit(inputs, outputs_, _units, activation, width, weight, method=method,
    #                               nbr_of_chunks=_nbr_of_chunks)
    #             fname = "./training-data/players/predicted-{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, _nbr_of_chunks, activation)
    #             utils.save(inputs, predictions, fname)
    #             print("========================================")


    # mixed x ~ f(x)_true ~ f(x)_estimated
    activation = "tanh"
    for method in methods:
        for weight in weights:
            for width in widths:
                print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))

                fname_true = "./training-data/players/{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, nbr_of_chunks, activation)
                fname_pred = "./training-data/players/predicted-{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, _nbr_of_chunks, activation)
                # fname_true = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)
                # fname_pred = "./training-data/players/predicted-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)

                inputs_true, outputs_true = utils.load(fname_true)
                inputs_pred, outputs_pred = utils.load(fname_pred)

                inputs = np.vstack([inputs_true, inputs_pred]).T
                outputs = np.vstack([outputs_true, outputs_pred]).T
                fname = "./pics/players/mixed-x-{}-{}-{}-{}-{}.gif".format(method, weight, width, units, activation)
                utils.save_animation(inputs, outputs, fname, colors=["blue", "red"], step=5)

    # mixed p ~ f(p)_true ~ f(p)_estimated
    # activation = "tanh"
    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname_operator = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #             fname_true = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)
    #             fname_pred = "./training-data/players/predicted-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)

    #             _, p = utils.load(fname_operator)
    #             _, outputs_true = utils.load(fname_true)
    #             _, outputs_pred = utils.load(fname_pred)

    #             inputs = np.vstack([p, p]).T
    #             outputs = np.vstack([outputs_true, outputs_pred]).T
    #             fname = "./pics/players/mixed-p-{}-{}-{}-{}-{}.gif".format(method, weight, width, units, activation)
    #             utils.save_animation(inputs, outputs, fname, colors=["blue", "red"], step=5)
