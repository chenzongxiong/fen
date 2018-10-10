import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from play import Phi, PlayCell

sess = tf.Session()


def generate_play_operator(weight=1.0, width=1.0, method="integer"):
    print("Processing method: {}, weeight: {}, width: {}".format(method, weight, width))
    if method == "integer":
        _inputs = np.random.uniform(low=-10, high=10, size=500)
    elif method == "float":
        _inputs = np.random.uniform(low=-10, high=10, size=500)
    else:
        raise
    np.insert(_inputs, 0, 0)              # set initial state as 0
    inputs = tf.constant(_inputs, dtype=tf.float32)

    pi_cell = PlayCell(weight=weight, width=width)
    outputs = pi_cell(inputs)
    inputs_res = sess.run(inputs)
    outputs_res = sess.run(outputs)

    res = np.vstack([inputs_res, outputs_res]).T
    np.savetxt("./training-data/operators/{}-{}-{}.csv".format(method, weight, width),
                res, fmt="%.3f.", delimiter=",", header="x,p")


    plt.figure()
    plt.scatter(inputs_res[1:], outputs_res[1:])
    fname = "./pics/operators/{}-{}-{}.pdf".format(method, weight, width)
    plt.savefig(fname, format="pdf")


if __name__ == "__main__":
    methods = ["integer", "float"]
    widths = [1, 2, 3, 4, 5, 5.5, 6.5, 7, 9, 10, 12, 20, 100]
    weights = [0.1, 1.0, 2.0, 3.0, 4.5]
    for method in methods:
        for weight in weights:
            for width in widths:
                generate_play_operator(weight, width, method)
