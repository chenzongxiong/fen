import os
import json
import numpy as np
import core
import utils
import colors
import log as logging
import constants

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
    def systhesis_sin_input_generator(cls, points, mu, sigma):
        # NOTE: x = sin(t) + 0.3 sin(1.3 t)  + 1.2 sin (1.6 t)
        inputs1 = np.sin(np.linspace(-2*np.pi, 2*np.pi, points))
        inputs2 = 0.3 * np.sin(1.3* np.linspace(-2*np.pi, 2*np.pi, points))
        inputs3 = 1.2 * np.sin(1.6 * np.linspace(-2*np.pi, 2*np.pi, points))
        inputs = (inputs1 + inputs2 + inputs3).astype(np.float32)
        LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = sin(t) + 0.3 sin(1.3 t)  + 1.2 sin (1.6 t)]")))

        noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
        inputs += noise
        return inputs

    @classmethod
    def systhesis_mixed_input_generator(cls, points, mu, sigma):
        # NOTE: x = cos(t) + 0.7 cos(3.0 t) + 1.5 sin(2.3 t)
        inputs1 = np.cos(np.linspace(-2*np.pi, 2*np.pi, points))
        inputs2 = 0.7 * np.cos(3.0 * np.linspace(-2*np.pi, 2*np.pi, points))
        inputs3 = 1.5 * np.sin(2.3 * np.linspace(-2*np.pi, 2*np.pi, points))
        inputs = (inputs1 + inputs2 + inputs3).astype(np.float32)
        LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = cos(t) + 0.7 cos(3.0 t)  + 1.5 sin (2.3 t)]")))

        noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
        inputs += noise
        return inputs

    @classmethod
    def systhesis_operator_generator(cls, points=1000, weight=1, width=1, state=0, with_noise=False, mu=0, sigma=0.01, method="sin"):
        if with_noise is True:
            if method == "sin":
                LOG.debug("Generate data with noise via sin method")
                _inputs = cls.systhesis_sin_input_generator(points, mu, sigma)
            elif method == "mixed":
                LOG.debug("Generate data with noise via mixed method")
                _inputs = cls.systhesis_mixed_input_generator(points, mu, sigma)
            elif method == "cos":
                LOG.debug("Generate data with noise via cos method")
                # _inputs = cls.systhesis_mixed_input_generator(points, mu, sigma)
                raise
        else:
            _inputs = cls.systhesis_input_generator(points)
        weight = float(weight)
        width = float(width)
        state = float(state)

        operator = core.Play(weight=weight,
                             width=width,
                             debug=True,
                             network_type=constants.NetworkType.OPERATOR)

        _outputs = operator.predict(_inputs)
        _outputs = _outputs.reshape(-1)
        return _inputs, _outputs

    @classmethod
    def systhesis_play_generator(cls, points=1000, inputs=None):
        if inputs is None:
            _inputs = cls.systhesis_input_generator(points)
        else:
            _inputs = inputs

        play = core.Play(debug=True,
                         network_type=constants.NetworkType.PLAY)

        _outputs = play.predict(_inputs)
        _outputs = _outputs.reshape(-1)
        return _inputs, _outputs

    @classmethod
    def systhesis_model_generator(cls, nb_plays=1, points=1000, units=1, debug_plays=False, inputs=None, batch_size=50):
        model = core.MyModel(nb_plays=nb_plays, units=units, debug=True, batch_size=batch_size)
        if inputs is None:
            # _inputs = cls.systhesis_input_generator(points)
            raise
        else:
            _inputs = inputs

        outputs = model.predict(_inputs)
        _outputs = outputs.reshape(-1)
        return _inputs, _outputs

    @staticmethod
    def systhesis_markov_chain_generator(points, mu, sigma, b0=0):
        B = [b0]
        for i in range(points-1):
            bi = np.random.normal(loc=B[-1] + mu, scale=sigma)
            B.append(bi)

        return np.array(B)


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
            elif d == 1:
                outputs = outputs.reshape(n,)
            elif d == inputs.shape[0]:
                outputs = outputs.T

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
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w") as f:
            f.write(json.dumps(loss))
