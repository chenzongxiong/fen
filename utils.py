import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def save(inputs, outputs, fname):
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(inputs.shape[0], 1)
    if len(outputs.shape) == 1:
        outputs = outputs.reshape(outputs.shape[0], 1)

    res = np.hstack([inputs, outputs])

    np.savetxt(fname, res, fmt="%.3f", delimiter=",")


def load_data(fname, split=1):
    data = np.loadtxt(fname, skiprows=0, delimiter=",", dtype=np.float32)
    inputs, outputs = data[:, 0], data[:, 1:].T
    assert len(inputs.shape) == 1
    if len(outputs.shape) == 2:
        n, d = outputs.shape
        if n == 1:
            outputs = outputs.reshape(d,)
        if d == 1:
            outputs = outputs.reshape(n,)
    if split == 1:
        return inputs, outputs

    split_index = int(split * inputs.shape[0])
    train_inputs, train_outputs = inputs[:split_index], outputs[:split_index]
    test_inputs, test_outputs = inputs[split_index:], outputs[split_index:]
    return (train_inputs, train_outputs), (test_inputs, test_outputs)


def update(i, *fargs):
    inputs = fargs[0]
    outputs = fargs[1]
    ax = fargs[2]
    colors = fargs[3]
    since = fargs[4]
    step = fargs[5]

    shape = inputs.shape


    if len(shape) == 1:
        if since is not None:
            idx = i // since
            if idx >= len(colors):
                idx = -1                      # force to always use the last color type
            ax.scatter(inputs[i:i+step], outputs[i:i+step], color=colors[idx])
        else:
            ax.scatter(inputs[i:i+step], outputs[i:i+step], color=colors[0])
    elif len(shape) == 2:
        if since is not None:
            ax.scatter(inputs[i:i+step, 0], outputs[i:i+step, 0], color=colors[0])
            ax.scatter(inputs[i:i+step, 1], outputs[i:i+step, 1], color=colors[1])
        else:
            ax.scatter(inputs[i:i+step, 0], outputs[i:i+step, 0], color=colors[0])
            ax.scatter(inputs[i:i+step, 1], outputs[i:i+step, 1], color=colors[1])


def save_animation(inputs, outputs, fname, xlim=None, ylim=None,
                   colors=["black"], step=1, since=None):
    if xlim is None:
        xlim = [np.min(inputs) - 1, np.max(inputs) + 1]
    if ylim is None:
        ylim = [np.min(outputs) - 1, np.max(outputs) + 1]

    assert inputs.shape == outputs.shape

    if not isinstance(colors, list):
        colors = [colors]

    fig, ax = plt.subplots(figsize=(20, 20))
    fig.set_tight_layout(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    points = inputs.shape[0]

    anim = FuncAnimation(fig, update, frames=np.arange(0, points, step),
                         fargs=(inputs, outputs, ax, colors, since, step), interval=300)
    anim.save(fname, dpi=40, writer='imagemagick')


_writer = None


def get_tf_summary_writer(fpath="."):
    global _writer
    if _writer is None:
        _writer = tf.summary.FileWriter(fpath)
    return _writer


writer = None
