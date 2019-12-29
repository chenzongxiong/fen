import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from fen import log as logging
from fen import constants
from fen import utils
from fen import colors
from fen.dense import MyDense, MySimpleDense
from fen.play import Play

LOG = logging.getLogger(__name__)


class BaseModel(object):
    def __init__(self,
                 inputs=None,
                 units=1,
                 batch_size=1,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 activation="tanh",
                 loss="mse",
                 optimizer="adam",
                 network_type=constants.NetworkType.OPERATOR,
                 use_bias=True,
                 name="play",
                 timestep=1,
                 input_dim=1,
                 **kwargs):

        np.random.seed(kwargs.pop("ensemble", 1))

        self._weight = weight
        self._width = width
        self._debug = debug

        self.activation = activation

        self.loss = loss
        self.optimizer = optimizer

        self._play_timestep = timestep
        self._play_batch_size = batch_size
        self._play_input_dim = input_dim

        self.units = units

        self._network_type = network_type
        self.built = False
        self._need_compile = False
        self.use_bias = use_bias
        self._name = name
        self._unittest = kwargs.pop('unittest', False)

    def _make_batch_input_shape(self, inputs=None):
        self._batch_input_shape = tf.TensorShape([1,
                                                  self._play_timestep,
                                                  self._play_input_dim])

    def build(self, inputs=None):
        if inputs is None and self._batch_input_shape is None:
            raise Exception("Unknown input shape")
        if inputs is not None:
            _inputs = ops.convert_to_tensor(inputs,
                                            dtype=tf.float32)

            if _inputs.shape.ndims == 1:
                length = _inputs.shape[-1].value
                if length % (self._play_input_dim * self._play_timestep) != 0:
                    LOG.error("length is: {}, input_dim: {}, play_timestep: {}".format(length,
                                                                                       self._play_input_dim,
                                                                                       self._play_timestep))
                    raise Exception("The batch size cannot be divided by the length of input sequence.")

                # self.batch_size = length // (self._play_timestep * self._play_input_dim)
                self._play_batch_size = length // (self._play_timestep * self._play_input_dim)
                self.batch_size = 1
                self._batch_input_shape = tf.TensorShape([self.batch_size, self._play_timestep, self._play_input_dim])

            else:
                raise Exception("dimension of inputs must be equal to 1")

        length = self._batch_input_shape[1].value * self._batch_input_shape[2].value
        self.batch_size = self._batch_input_shape[0].value
        assert self.batch_size == 1, colors.red("only support batch_size is 1")
        if not getattr(self, "_unittest", False):
            assert self._play_timestep == 1, colors.red("only support outter-timestep 1")

        self.model = tf.keras.models.Sequential()

        CACHE = utils.get_cache()
        input_layer = CACHE.get('play_input_layer', None)

        if input_layer is None:
            input_layer = tf.keras.layers.InputLayer(batch_size=self.batch_size,
                                                     input_shape=self._batch_input_shape[1:])
            CACHE['play_input_layer'] = input_layer

        self.model.add(input_layer)

        self.model.add(Play(weight=getattr(self, "_weight", 1.0),
                            width=getattr(self, "_width", 1.0),
                            debug=getattr(self, "_debug", False),
                            unittest=getattr(self, "_unittest", False)))

        if self._network_type == constants.NetworkType.PLAY:
            self.model.add(MyDense(self.units,
                                   activation=self.activation,
                                   use_bias=self.use_bias,
                                   debug=getattr(self, "_debug", False)))

            self.model.add(MySimpleDense(units=1,
                                         activation=None,
                                         use_bias=True,
                                         debug=getattr(self, "_debug", False)))

        if self._need_compile is True:
            LOG.info(colors.yellow("Start to compile this model"))
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=[self.loss])

        self._early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)
        self._tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=constants.LOG_DIR,
                                                                     histogram_freq=0,
                                                                     batch_size=self.batch_size,
                                                                     write_graph=True,
                                                                     write_grads=False,
                                                                     write_images=False)

        # if not getattr(self, "_preload_weights", False):
        #     utils.init_tf_variables()
        if self._network_type == constants.NetworkType.OPERATOR:
            LOG.debug(colors.yellow("SUMMARY of Operator"))
        elif self._network_type == constants.NetworkType.PLAY:
            LOG.debug(colors.yellow("SUMMARY of {}".format(self._name)))
        else:
            raise
        self.model.summary()
        self.built = True

    def reshape(self, inputs, outputs=None):
        LOG.debug("reshape inputs to: {}".format(self._batch_input_shape))
        x = tf.reshape(inputs, shape=self._batch_input_shape)
        if outputs is not None:
            if self._network_type == constants.NetworkType.OPERATOR:
                y = tf.reshape(outputs, shape=(self._batch_input_shape[0].value, -1, 1))
            elif self._network_type == constants.NetworkType.PLAY:
                y = tf.reshape(outputs, shape=(self._batch_input_shape[0].value, -1, 1))
            return x, y
        else:
            return x

    def fit(self, inputs, outputs, epochs=100, verbose=0, steps_per_epoch=1):
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        outputs = ops.convert_to_tensor(outputs, tf.float32)
        self._need_compile = True
        if not self.built:
            self.build(inputs)

        x, y = self.reshape(inputs, outputs)

        self.model.fit(x,
                       y,
                       epochs=epochs,
                       verbose=verbose,
                       steps_per_epoch=steps_per_epoch,
                       batch_size=None,
                       shuffle=False,
                       callbacks=[self._early_stopping_callback,
                                  self._tensor_board_callback])

    def evaluate(self, inputs, outputs, steps_per_epoch=1):
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        outputs = ops.convert_to_tensor(outputs, tf.float32)

        if not self.built:
            self.build(inputs)

        x, y = self.reshape(inputs, outputs)
        return self.model.evaluate(x, y, steps=steps_per_epoch)

    def predict(self, inputs, steps_per_epoch=1, verbose=0, states=None):
        if not self.built:
            self.build(inputs)

        outputs = []
        input_dim = self._batch_input_shape[-1].value
        samples = inputs.shape[-1] // input_dim
        LOG.debug("#Samples: {}".format(samples))
        for j in range(samples):
            # LOG.debug("PID: {}, self.states: {}, states: {} before".format(os.getpid(), sess.run(self.states), states))
            self.reset_states(states)
            # LOG.debug("PID: {}, self.states: {}, states: {} after".format(os.getpid(), sess.run(self.states), states))
            x = inputs[j*input_dim:(j+1)*input_dim].reshape(1, 1, -1)
            output = self.model.predict(x, steps=steps_per_epoch, verbose=verbose).reshape(-1)
            # LOG.debug("PID: {}, self.states: {}, states: {} done".format(os.getpid(), sess.run(self.states), states))
            outputs.append(output)
            if j != samples - 1:
                op_output = self.operator_output(x, states)
                states = op_output[-1].reshape(1, 1)

        return np.hstack(outputs)

    @property
    def states(self):
        return self.operator_layer.states

    @property
    def weights(self):
        if self._network_type == constants.NetworkType.OPERATOR:
            weights_ = [self.operator_layer.kernel]
        elif self._network_type == constants.NetworkType.PLAY:
            weights_ = [self.operator_layer.kernel,
                        self.nonlinear_layer.kernel,
                        self.nonlinear_layer.bias,
                        self.linear_layer.kernel,
                        self.linear_layer.bias]
        else:
            raise Exception("Unknown NetworkType. It must be in [OPERATOR, PLAY]")

        weights_ = utils.get_session().run(weights_)
        weights_ = [w.reshape(-1) for w in weights_]
        return weights_

    def operator_output(self, inputs, states=None):
        if len(inputs.shape) == 1:
            input_dim = self._batch_input_shape[-1].value
            samples = inputs.shape[-1] // input_dim
        elif list(inputs.shape) == self._batch_input_shape.as_list():
            input_dim = inputs.shape[-1]
            samples = inputs.shape[0]
        else:
            raise Exception("Unknown input.shape: {}".format(inputs.shape))

        outputs = []
        for j in range(samples):
            self.reset_states(states)
            x = inputs[j*input_dim:(j+1)*input_dim].reshape(1, 1, -1)
            op_output = self.operator_layer(ops.convert_to_tensor(x, dtype=tf.float32))
            output = utils.get_session().run(op_output).reshape(-1)
            outputs.append(output)
            states = output[-1].reshape(1, 1)

        outputs = np.hstack(outputs)
        return outputs.reshape(-1)

    def reset_states(self, states=None):
        self.operator_layer.reset_states(states)

    @property
    def number_of_layers(self):
        if not self.built:
            raise Exception("Model has not been built.")

        return len(self.model._layers)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path, by_name=False):
        self.model.load_weights(path, by_name=by_name)
        # self.model.load_weights(path, by_name=True)

    @property
    def layers(self):
        if hasattr(self, 'model'):
            return self.model.layers
        return []

    @property
    def _layers(self):
        if hasattr(self, 'model'):
            return self.model.layers
        return []

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def output(self):
        return self.model._layers[-1].output

    @property
    def input(self):
        return self.model._layers[0].input

    @property
    def state_updates(self):
        return self.model.state_updates

    @property
    def operator_layer(self):
        return self.layers[0]

    @property
    def nonlinear_layer(self):
        return self.layers[1]

    @property
    def linear_layer(self):
        return self.layers[2]

    def __getstate__(self):
        LOG.debug("PID: {} pickle {}".format(os.getpid(), self._name))
        if not hasattr(self, '_batch_input_shape'):
            raise Exception("_batch_input_shape must be added before pickling")

        state = {
            "_weight": self._weight,
            "_width": self._width,
            "_debug": self._debug,
            "activation": self.activation,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "_play_timestep": self._play_timestep,
            "_play_batch_size": self._play_batch_size,
            "_play_input_dim": self._play_input_dim,
            "units": self.units,
            "_network_type": self._network_type,
            "_built": getattr(self, "_built", False),
            "_need_compile": self._need_compile,
            "use_bias": self.use_bias,
            "_name": self._name,
            "_weights_fname": getattr(self, '_weights_fname', None),
            "_preload_weights": getattr(self, '_preload_weights', False),
            "_batch_input_shape": getattr(self, '_batch_input_shape'),
        }
        return state

    def __setstate__(self, d):
        LOG.debug("PID: {}, unpickle {}".format(os.getpid(), colors.cyan(d)))
        self.__dict__ = d
        if self._built is False:
            self.build()
        if self._preload_weights is False and self._weights_fname is not None:
            self._preload_weights = True
            self.load_weights(self._weights_fname)
            LOG.debug(colors.cyan("Set weights to play in sub-process"))

        LOG.debug("PID: {}, self: {}, self.model: {}".format(os.getpid(), self, self.model))

    def __hash__(self):
        return hash(self._name)
