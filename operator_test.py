import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from play import Phi, PlayCell
import utils

sess = tf.Session()


def generator(weight=1.0, width=1.0, method="integer", state=0):
    points = 200
    if method == "integer":
        _inputs = np.random.uniform(low=-10, high=10, size=points)
    elif method == "float":
        _inputs = np.random.uniform(low=-10, high=10, size=points)
    elif method == "sin":
        _inputs1 = np.sin(np.linspace(-2*np.pi, 2*np.pi, points))
        _inputs2 = 3 * np.sin(1.3* np.linspace(-2*np.pi, 2*np.pi, points))
        _inputs3 = 1.2 * np.sin(1.6 * np.linspace(-2*np.pi, 2*np.pi, points))
        _inputs = _inputs1 + _inputs2 + _inputs3

    inputs = tf.constant(_inputs, dtype=tf.float32)

    cell = PlayCell(weight=weight, width=width, debug=True)

    outputs = cell(inputs, state)

    init = tf.global_variables_initializer()
    sess.run(init)

    inputs_res = sess.run(inputs)
    outputs_res = sess.run(outputs)

    return inputs_res, outputs_res


def fit(inputs, outputs, width, method, true_weight):

    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)
    cell = PlayCell(width=width, debug=False)

    predictions = cell(inputs, state)
    loss = tf.losses.mean_squared_error(labels=outputs,
                                        predictions=predictions)

    loss_summary = tf.summary.scalar("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)
    epochs = 500

    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        if i % 10 == 0:         # every 10 times
            summary = sess.run(loss_summary)
            utils.writer.add_summary(summary, i)

        print("epoch", i, "loss:", loss_value, ", weight: ", cell.kernel.eval(session=sess))

    predictions = cell(inputs, 0)
    return predictions


if __name__ == '__main__':
    utils.writer = utils.get_tf_summary_writer('./log/operator')

    save_anim = False
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_anim = True

    states = [0, -1, 3, 7, 30, -30]
    methods = ["sin"]
    widths = [5]
    weights = [2]

    # # save generated dataset
    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             inputs = []
    #             outputs = []
    #             for state in states:
    #                 print("Processing method: {}, weight: {}, width: {}, state: {}".format(method, weight, width, state))
    #                 inputs_, outputs_ = generator(weight, width, method, state)
    #                 inputs.append(inputs_)
    #                 outputs.append(outputs_)
    #             fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #             inputs = np.hstack(inputs)
    #             outputs = np.hstack(outputs)
    #             utils.save(inputs, outputs, fname)


    # # save animation, plot
    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #             inputs, outputs = utils.load(fname)
    #             fname = "./pics/operators/{}-{}-{}.gif".format(method, weight, width)

    #             utils.save_animation(inputs, outputs, fname, step=10, since=100, colors=["black", "red", "blue"])
    #             print("Finished ", fname)

    # #  train dataset
    for method in methods:
        for weight in weights:
            for width in widths:
                print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                inputs, outputs = utils.load(fname)
                fit(inputs, outputs, width, method, weight)
                print("========================================")
