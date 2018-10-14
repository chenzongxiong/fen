import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from play import Phi, PlayCell

sess = tf.Session()


def generate_play_operator(weight=1.0, width=1.0, method="integer"):
    print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    points = 100
    if method == "integer":
        _inputs = np.random.uniform(low=-10, high=10, size=points)
    elif method == "float":
        _inputs = np.random.uniform(low=-10, high=10, size=points)
    elif method == "sin":
        _inputs1 = 10 * np.sin(np.linspace(-np.pi, np.pi, points))
        _inputs2 = 5 * np.sin(np.linspace(-np.pi, np.pi, points))
        _inputs = np.hstack([_inputs1, _inputs2])

    inputs = tf.constant(_inputs, dtype=tf.float32)

    pi_cell = PlayCell(weight=weight, width=width, debug=True)
    outputs = pi_cell(inputs, 0)
    inputs_res = sess.run(inputs)
    outputs_res = sess.run(outputs)

    return inputs_res, outputs_res


def save(inputs, outputs, fname):
    res = np.vstack([inputs, outputs]).T
    np.savetxt(fname, res, fmt="%.3f", delimiter=",", header="x,p")


def load(fname):
    data = np.loadtxt(fname, skiprows=1, delimiter=",", dtype=np.float32)
    inputs, outputs = data[:, 0], data[:, 1]
    return inputs, outputs


def update(i, *fargs):
    inputs = fargs[0]
    outputs = fargs[1]
    ax = fargs[-1]
    ax.scatter(inputs[:i], outputs[:i], color="black")


def save_animation(inputs, outputs, fname, xlim, ylim):
    fig, ax = plt.subplots(figsize=(20, 20))
    fig.set_tight_layout(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    points = inputs.shape[0]
    anim = FuncAnimation(fig, update, frames=np.arange(0, points), fargs=(inputs, outputs, ax), interval=300)
    anim.save(fname, dpi=40, writer='imagemagick')


def fit(inputs, outputs, width, method, true_weight):
    print("Processing method: {}, weight: {}, width: {}".format(method, true_weight, width))

    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)
    pi_cell = PlayCell(width=width, debug=False)

    predictions = pi_cell(inputs, state)
    loss = tf.losses.mean_squared_error(labels=outputs,
                                        predictions=predictions)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)
    epochs = 30
    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        print("epoch", i, "loss:", loss_value, ", weight: ", pi_cell.kernel.eval(session=sess))

    with open("./training-data/operators/trained-{}-{}-{}.txt".format(method, true_weight, width), "w") as f:
        print("true weight: {}, estimated_weight: {}".format(true_weight, pi_cell.kernel.eval(session=sess)))
        f.write("true_weight: {}, estimated_weight: {}".format(true_weight, pi_cell.kernel.eval(session=sess)))

    outputs_ = pi_cell(inputs, state)
    fname = "./training-data/operators/pred-{}-{}-{}.csv".format(method, true_weight, width)
    save(inputs, outputs_.eval(session=sess), fname)

    # import ipdb; ipdb.set_trace()
    print("========================================")


if __name__ == '__main__':
    save_anim = False
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_anim = True

    methods = ["integer", "float"]
    widths = [1, 2, 3, 4, 5, 5.5, 6.5, 7, 9, 10, 12, 20, 100]
    weights = [1.0, 2.0, 3.0, 4.5]

    for method in methods:
        for weight in weights:
            for width in widths:
                inputs, outputs = generate_play_operator(weight, width, method)
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                save(inputs, outputs, fname)
                # if save_anim is True:
                #     fname = "./pics/operators/{}-{}-{}.gif".format(method, weight, width)
                #     xlim = [np.min(inputs) - 1, np.max(inputs) + 1]
                #     ylim = [np.min(outputs) - 1, np.max(outputs) + 1]
                #     save_animation(inputs, outputs, fname, xlim, ylim)
                #     print("Finished ", fname)
                # else:
                #     plt.show()

    for method in methods:
        for weight in weights:
            for width in widths:
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                inputs, outputs = load(fname)
                fit(inputs, outputs, width, method, weight)
