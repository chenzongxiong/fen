import sys
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from play import Phi, PlayCell
from matplotlib.animation import FuncAnimation


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

    _inputs = np.insert(_inputs, 0, 0)              # set initial state as 0
    points += 1
    inputs = tf.constant(_inputs, dtype=tf.float32)

    pi_cell = PlayCell(weight=weight, width=width)
    outputs = pi_cell(inputs)
    inputs_res = sess.run(inputs)
    outputs_res = sess.run(outputs)

    return inputs_res, outputs_res


def save(inputs, outputs, method, weight, width):
    res = np.vstack([inputs, outputs]).T
    np.savetxt("./training-data/operators/{}-{}-{}.csv".format(method, weight, width),
                res, fmt="%.3f", delimiter=",", header="x,p")

    # plt.scatter(inputs, outputs)
    # fname = "./pics/operators/{}-{}-{}.pdf".format(method, weight, width)

    # plt.figure(figsize=(9, 9))
    # plt.xlim(-10, 10)
    # plt.ylim(-20, 20)
    # plt.savefig(fname, format="pdf")


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


if __name__ == '__main__':
    save_anim = False
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_anim = True

    methods = ["sin"]
    widths = [2, 3, 4, 5, 5.5, 6.5, 7, 9, 10, 12, 20, 100]
    weights = [1.0, 2.0, 3.0, 4.5]

    for method in methods:
        for weight in weights:
            for width in widths:
                inputs, outputs = generate_play_operator(weight, width, method)
                save(inputs, outputs, method, weight, width)
                if save_anim is True:
                    fname = "./pics/operators/{}-{}-{}.gif".format(method, weight, width)
                    xlim = [np.min(inputs) - 1, np.max(inputs) + 1]
                    # xlim = [-15, 15]
                    ylim = [np.min(outputs) - 1, np.max(outputs) + 1]
                    # ylim = [-20, 20]
                    save_animation(inputs, outputs, fname, xlim, ylim)
                    print("Finished ", fname)
                else:
                    plt.show()
