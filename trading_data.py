import numpy as np
import pandas as pd
import tensorflow as tf
import play
import model
import utils
import colors
import log as logging


LOG = logging.getLogger(__name__)
CSV_COLUMN_NAMES = ["x", "y"]


class DatasetGenerator():
    @classmethod
    def systhesis_input_generator(cls, points):
        # NOTE: x = sin(t) + 3 sin(1.3 t)  + 1.2 sin (1.6 t)
        inputs1 = np.sin(np.linspace(-2*np.pi, 2*np.pi, points))
        inputs2 = 3 * np.sin(1.3* np.linspace(-2*np.pi, 2*np.pi, points))
        inputs3 = 1.2 * np.sin(1.6 * np.linspace(-2*np.pi, 2*np.pi, points))
        inputs = (inputs1 + inputs2 + inputs3).astype(np.float32)
        LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = sin(t) + 3 sin(1.3 t)  + 1.2 sin (1.6 t)]")))
        return inputs

    @classmethod
    def systhesis_play_operator_generator(cls, points=1000, weight=1, width=1, state=0):
        _inputs = cls.systhesis_input_generator(points)
        weight = float(weight)
        width = float(width)
        state = float(state)

        cell = play.PlayCell(weight=weight, width=width, debug=True)
        outputs = cell(_inputs, state)
        sess = utils.get_session()
        _outputs = sess.run(outputs)
        return _inputs, _outputs

    @classmethod
    def systhesis_play_generator(cls, points=1000):
        _inputs = cls.systhesis_input_generator(points)
        cell = play.PlayCell(debug=True)
        layer = play.Play(units=4,
                          cell=cell,
                          debug=True)
        outputs = layer(_inputs)
        sess = utils.get_session()
        _outputs = sess.run(outputs)
        return _inputs, _outputs

    @classmethod
    def systhesis_model_generator(cls, nb_plays=1, points=1000, debug_plays=False):
        play_model = model.PlayModel(nb_plays=nb_plays, debug=True)
        _inputs = cls.systhesis_input_generator(points)
        plays_outputs = play_model.get_plays_outputs(_inputs.reshape(1, -1))

        if debug_plays is True:
            return _inputs, plays_outputs.sum(axis=1), plays_outputs
        else:
            return _inputs, plays_outputs.sum(axis=1)



def load_data(fname):
    # data = np.loadtxt(fname, skiprows=0, delimiter=",", dtype=np.float32)
    # inputs, outputs = data[:, 0], data[:, 1:].T
    # assert len(inputs.shape) == 1
    # if len(outputs.shape) == 2:
    #     n, d = outputs.shape
    #     if n == 1:
    #         outputs = outputs.reshape(d,)
    #     if d == 1:
    #         outputs = outputs.reshape(n,)
    data = pd.read_csv(fname, names=CSV_COLUMN_NAMES, header=0)
    inputs, outputs = data, data.pop("y")
    # inputs = inputs.reshape(1, inputs.shape[0])
    # outputs = outputs.reshape(1, outputs.shape[0])
    return inputs, outputs


def train_input_fn(inputs, outputs, batch_size):
    # dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), outputs))
    # dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    inputs = inputs.values
    inputs = inputs.reshape(1, inputs.shape[0])
    outputs = outputs.values
    outputs = outputs.reshape(1, outputs.shape[0])

    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

    import ipdb; ipdb.set_trace()
    # dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.repeat().batch(batch_size)
    return dataset
