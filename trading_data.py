import os
import json
import numpy as np
import core
import utils
import colors
import log as logging


LOG = logging.getLogger(__name__)


class DatasetGenerator(object):
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
    def systhesis_operator_generator(cls, points=1000, weight=1, width=1, state=0):
        _inputs = cls.systhesis_input_generator(points)
        weight = float(weight)
        width = float(width)
        state = float(state)

        cell = core.PlayCell(weight=weight, width=width, debug=True)
        outputs = cell(_inputs, state)
        sess = utils.get_session()
        _outputs = sess.run(outputs)
        return _inputs, _outputs

    @classmethod
    def systhesis_play_generator(cls, points=1000, inputs=None):
        if inputs is None:
            _inputs = cls.systhesis_input_generator(points)
        else:
            _inputs = inputs

        cell = core.PlayCell(debug=True)
        layer = core.Play(units=4,
                          cell=cell,
                          debug=True)
        outputs = layer(_inputs)
        sess = utils.get_session()
        _outputs = sess.run(outputs)
        return _inputs, _outputs

    @classmethod
    def systhesis_model_generator(cls, nb_plays=1, points=1000, debug_plays=False, inputs=None):
        play_model = core.PlayModel(nb_plays=nb_plays, debug=True)
        if inputs is None:
            _inputs = cls.systhesis_input_generator(points)
        else:
            _inputs = inputs

        plays_outputs = play_model.get_plays_outputs(_inputs.reshape(1, -1))

        if debug_plays is True:
            return _inputs, plays_outputs.sum(axis=1), plays_outputs
        else:
            return _inputs, plays_outputs.sum(axis=1)


class DatasetLoader(object):
    SPLIT_RATIO = 0.6
    _CACHED_DATASET = {}

    @classmethod
    def load_data(cls, fname):
        if fname in cls._CACHED_DATASET:
            return cls._CACHED_DATASET[fname]

        data = np.loadtxt(fname, skiprows=0, delimiter=",", dtype=np.float32)
        inputs, outputs = data[:, 0], data[:, 1:].T
        assert len(inputs.shape) == 1
        if len(outputs.shape) == 2:
            n, d = outputs.shape
            if n == 1:
                outputs = outputs.reshape(d,)
            if d == 1:
                outputs = outputs.reshape(n,)

        cls._CACHED_DATASET[fname] = (inputs, outputs)
        return inputs, outputs

    @classmethod
    def load_train_data(cls, fname):
        if fname in cls._CACHED_DATASET:
            inputs, outputs = cls._CACHED_DATASET[fname]
            LOG.debug("Load train dataset {} from cache".format(colors.red(fname)))
        else:
            inputs, outputs = cls.load_data(fname)
            cls._CACHED_DATASET[fname] = (inputs, outputs)

        split_index = int(cls.SPLIT_RATIO * inputs.shape[0])
        train_inputs, train_outputs = inputs[:split_index], outputs[:split_index]
        return train_inputs, train_outputs

    @classmethod
    def load_test_data(cls, fname):
        if fname in cls._CACHED_DATASET:
            inputs, outputs = cls._CACHED_DATASET[fname]
            LOG.debug("Load test dataset {} from cache".format(colors.red(fname)))
        else:
            inputs, outputs = cls.load_data(fname)
            cls._CACHED_DATASET[fname] = (inputs, outputs)

        split_index = int(cls.SPLIT_RATIO * inputs.shape[0])
        test_inputs, test_outputs = inputs[split_index:], outputs[split_index:]
        return test_inputs, test_outputs


class DatasetSaver(object):
    @staticmethod
    def save_data(inputs, outputs, fname):
        assert len(inputs.shape) == 1, "length of inputs.shape must be equal to 1."
        assert inputs.shape[0] == outputs.shape[0], \
          "inputs.shape[0] is: {}, whereas outputs.shape[0] is {}.".format(inputs.shape[0], outputs.shape[0])
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)
        if len(outputs.shape) == 1:
            outputs = outputs.reshape(-1, 1)

        res = np.hstack([inputs, outputs])
        np.savetxt(fname, res, fmt="%.3f", delimiter=",")


    @staticmethod
    def save_loss(loss, fname):
        with open(fname, "w") as f:
            f.write(json.dumps(loss))
