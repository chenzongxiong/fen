import os
import threading

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import log as logging
import colors

LOG = logging.getLogger(__name__)


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

    if mode == "sequence":
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+step, x], outputs[i:i+step, x], color=colors[x])
    elif mode == "snake":
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+10, x], outputs[i:i+10, x], color=colors[x])


def save_animation(inputs, outputs, fname, xlim=None, ylim=None,
                   colors=["black"], step=1, mode="sequence"):
    assert inputs.shape == outputs.shape
    assert mode in ["sequence", "snake"], "mode must be 'sequence' or 'snake'."
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    if xlim is None:
        xlim = [np.min(inputs) - 1, np.max(inputs) + 1]
    if ylim is None:
        ylim = [np.min(outputs) - 1, np.max(outputs) + 1]

    if len(inputs.shape) == 1:
        inputs = inputs.reshape(-1, 1)
        outputs = outputs.reshape(-1, 1)
    if not isinstance(colors, list):
        colors = [colors]

    assert len(colors) == outputs.shape[1]

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
        LOG.error(colors.red("Doesn't have enough colors"))
        raise
    return COLORS[:length]


class TFSummaryFileWriter(object):
    _writer = None
    _lock = threading.Lock()

    def __new__(cls, fpath="."):
        import tensorflow as tf

        if not cls._writer:
            with cls._lock:
                if not cls._writer:
                    cls._writer = tf.summary.FileWriter(fpath)
        return cls._writer


def get_tf_summary_writer(fpath):
    writer = TFSummaryFileWriter(fpath)
    return writer


def get_session():
    import tensorflow as tf
    return tf.keras.backend.get_session()


def init_tf_variables():
    import tensorflow as tf
    sess = tf.keras.backend.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
