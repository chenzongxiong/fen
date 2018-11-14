import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import log as logging

LOG = logging.getLogger(__name__)


def save_data(inputs, outputs, fname):
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
    mode = fargs[4]
    step = fargs[5]
    if mode == "snake":
        xlim = fargs[6]
        ylim = fargs[7]
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if i % 100 == 0:
        LOG.info("Update animation frame: {}, step: {}".format(i, step))

    shape = inputs.shape
    if mode == "sequence":
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+step, x], outputs[i:i+step, x], color=colors[x])
    elif mode == "snake":
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+10, x], outputs[i:i+10, x], color=colors[x])


def save_animation(inputs, outputs, fname, xlim=None, ylim=None,
                   colors=["black"], step=1, mode="sequence"):
    assert inputs.shape == outputs.shape
    assert mode in ["sequence", "snake"]

    if xlim is None:
        xlim = [np.min(inputs) - 1, np.max(inputs) + 1]
    if ylim is None:
        ylim = [np.min(outputs) - 1, np.max(outputs) + 1]

    if len(inputs.shape) == 1:
        inputs = inputs.reshape(-1, 1)
        outputs = outputs.reshape(-1, 1)

    if not isinstance(colors, list):
        colors = [colors]

    assert len(colors) == inputs.shape[1]

    fig, ax = plt.subplots(figsize=(20, 20))
    fig.set_tight_layout(True)
    points = inputs.shape[0]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fargs=(inputs, outputs, ax, colors, mode, step, xlim, ylim)

    anim = FuncAnimation(fig, update, frames=np.arange(0, points, step),
                         fargs=fargs, interval=300)
    anim.save(fname, dpi=40, writer='imagemagick')


COLORS = ["blue", "black", "orange", "cyan", "red", "magenta", "yellow", "green"]

def generate_colors(length=1):
    if (length >= len(COLORS)):
        LOG.error("Doesn't have enough colors")
        raise
    return COLORS[:length]

# TODO: test tf summary writer
class TFSummaryFileWriter(object):
    _writer = None
    def __new__(cls, fpath="."):
        if not cls._writer:
            cls._writer = tf.summary.FileWriter(fpath)
        return cls._writer

writer = TFSummaryFileWriter()

def get_tf_summary_writer():
    global writer
    return writer


def get_session():
    return tf.keras.backend.get_session()
