import os
import threading
import h5py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import log as logging
import colors

LOG = logging.getLogger(__name__)
# base seed
# np.random.seed(123)
# test seed
# np.random.seed(345)
# LOG.debug(colors.red("Make sure you are using the right random seed. currently seed is 345"))
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())


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

    s = [n*8 for n in range(step)]
    if mode == "sequence":
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+step, x], outputs[i:i+step, x], color=colors[x])
    elif mode == "snake":
        inputs_len = inputs.shape[0] // 2
        for x in range(len(colors)):
            ax.scatter(inputs[0:inputs_len, x], outputs[0:inputs_len, x], color='cyan')
        for x in range(len(colors)):
            ax.scatter(inputs[i+inputs_len:i+step+inputs_len, x], outputs[i+inputs_len:i+step+inputs_len, x], color=colors[x], s=s)


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

    anim = None
    if mode == "sequence":
        anim = FuncAnimation(fig, update, frames=np.arange(0, points, step),
                             fargs=fargs, interval=400)
    elif mode == "snake":
        frame_step = step // 4
        if frame_step == 0:
            frame_step = 2
        # frame_step = 2
        anim = FuncAnimation(fig, update, frames=np.arange(0, points // 2, frame_step),
                             fargs=fargs, interval=400)

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


_SESSION = None

def get_session(debug=False, interactive=False):
    # https://github.com/tensorflow/tensorflow/issues/5448
    # import multiprocessing as mp
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    import tensorflow as tf
    from tensorflow.python import debug as tf_debug

    global _SESSION
    if _SESSION is not None:
        return _SESSION

    if debug is True:
        _SESSION = tf.keras.backend.set_session(
            tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:1234"))
    elif interactive is True:
        _SESSION = tf.InteractiveSession()
    else:
        _SESSION = tf.keras.backend.get_session()

    return _SESSION


def init_tf_variables():
    import tensorflow as tf
    sess = tf.keras.backend.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)


def read_saved_weights(fname=None):
    f = h5py.File(fname, 'r')

    for k in list(f.keys())[::-1]:
        for kk in list(f[k].keys())[::-1]:
            for kkk in list(f[k][kk].keys())[::-1]:
                print("Layer *{}*, {}: {}".format(colors.red(kk.upper()), colors.red(kkk), list(f[k][kk][kkk])))
    f.close()


def build_play(play, inputs):
    if not play.built:
        play.build(inputs)
    return play


def build_p3(p2, j):
    import tensorflow as tf
    return tf.reduce_sum(tf.cumprod(p2[:, j:], axis=1), axis=1)


def slide_window_average(arr, window_size=5, step=1):
    assert len(arr.shape) == 1, colors.red("slide window only support 1-dim")
    if window_size == 1:
        return arr

    N = arr.shape[0]
    stacked_array = np.vstack([ arr[i: 1 + N + i - window_size:step] for i in range(window_size) ])
    avg = np.concatenate([stacked_array.mean(axis=0), arr[-window_size+1:]])
    return avg


def _parents(op):
  return set(input.op for input in op.inputs)

def parents(op, indent=0, ops_have_been_seen=set()):
    # for op in ops:
    #     _ops = list(set(input.op for input in op.inputs))
    #     print("_ops: {}".format(_ops))]
    #     parents(_ops, indent=indent+1, ops_have_been_seen=ops_have_been_seen)
    print(len(op.inputs))
    for inp in op.inputs:
        print(op.name, '-->', inp.op.name)
        parents(inp.op)

    print("Indent: {}".format(indent))


def children(op):
  return set(op for out in op.outputs for op in out.consumers())

def get_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph. Result is compatible with networkx/toposort"""
  import tensorflow as tf
  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}



def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""
  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))
    print("--------------------------------------------------------------------------------")

import matplotlib.pyplot as plt
import networkx as nx
def plot_graph(G):
    '''Plot a DAG using NetworkX'''
    def mapping(node):
        return node.name
    G = nx.DiGraph(G)
    nx.relabel_nodes(G, mapping, copy=False)
    nx.draw(G, cmap = plt.get_cmap('jet'), with_labels=True)
    plt.show()


def plot_hysteresis_info(hysteresis_info, i=None, predicted_price=None):
    fig = plt.figure()
    colors = ['magenta']

    for index, info in enumerate(hysteresis_info):
        guess_hysteresis_list = info[0]
        prices = np.array([g[0] for g in guess_hysteresis_list])
        noise = np.array([g[1] for g in guess_hysteresis_list])
        l = len(prices)
        prev_original_prediction = np.array([info[1]] * l)
        curr_original_prediction = np.array([info[2]] * l)
        bk = np.array([info[3]] * l)
        prev_price = np.array([info[4]] * l)
        curr_price = np.array([info[5]] * l)
        _predicted_price = np.array([predicted_price] * l)
        vertical_line = np.linspace(noise.min(), noise.max(), l)
        if index == 0:
            plt.plot(prices, prev_original_prediction, color='red', label='start position', linewidth=1)
            plt.plot(prices, bk, color='black', label='random walk', linewidth=1)
            plt.plot(prices, curr_original_prediction, color='blue', label='target position', linewidth=1)
            plt.plot(prev_price, vertical_line, color='red', label='previous price', linewidth=1, linestyle='dashed')
            plt.plot(curr_price, vertical_line, color='blue', label='current price', linewidth=1, linestyle='dashed')
            plt.plot(_predicted_price, vertical_line, color='black', label='predicted price', linewidth=1, linestyle='dashed')
            plt.plot(prices, noise, color=colors[index % len(colors)], marker='o', markersize=4, label='steps finding root', linewidth=1)
        else:
            plt.plot(prices, prev_original_prediction, color='red', label=None, linewidth=1)
            plt.plot(prices, bk, color='black', label=None, linewidth=1)
            plt.plot(prices, curr_original_prediction, color='blue', label=None, linewidth=1)
            plt.plot(prev_price, vertical_line, color='red', label=None, linewidth=1, linestyle='dashed')
            plt.plot(curr_price, vertical_line, color='blue', label=None, linewidth=1, linestyle='dashed')
            plt.plot(_predicted_price, vertical_line, color='black', label=None, linewidth=1, linestyle='dashed')
            plt.plot(prices, noise, color=colors[index % len(colors)], marker='o', markersize=4, label=None, linewidth=1)

    plt.xlabel("prices")
    plt.ylabel("noise")
    plt.legend()
    # plt.show()
    fname = './frames/{}.png'.format(i)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=400)
    LOG.debug("Save picture into disk {}".format(fname))


if __name__ == "__main__":
    import numpy as np
    arr = np.arange(100)
    # arr1 = slide_window_average(arr, 1)
    # arr2 = slide_window_average(arr, 2)
    arr3 = slide_window_average(arr, 3)
