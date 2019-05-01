import os
import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers import Dense

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import training_arrays
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops

import numpy as np

import utils
import colors
import constants
import trading_data as tdata
import log as logging


LOG = logging.getLogger(__name__)

sess = utils.get_session()
session = utils.get_session()
SESS = utils.get_session()
SESSION = utils.get_session()


tf.keras.backend.set_epsilon(1e-7)


def Phi(x, width=1.0):
    '''
    Phi(x) = x - width/2 , if x > width/2
           = x + width/2 , if x < - width/2
           = 0         , otherwise
    '''
    assert x.shape[0].value == 1 and x.shape[1].value == 1, "x must be a scalar"

    ZEROS = tf.zeros(x.shape, dtype=tf.float32, name='zeros')
    _width = tf.constant([[1/2.0]], dtype=tf.float32)

    r1 = tf.cond(tf.reduce_all(tf.less(x, -_width)), lambda: x + _width, lambda: ZEROS)
    r2 = tf.cond(tf.reduce_all(tf.greater(x, _width)), lambda: x - _width, lambda: r1)
    return r2


def gradient_operator(P, weights=None):
    # _P = tf.reshape(P, shape=(P.shape[0].value, -1))
    # _diff = _P[:, 1:] - _P[:, :-1]

    # x0 = tf.slice(_P, [0, 0], [1, 1])
    # diff = tf.concat([x0, _diff], axis=1)

    # p1 = tf.cast(tf.abs(diff) > 0., dtype=tf.float32)
    # p2 = 1.0 - p1
    # p3_list = []
    # # TODO: multiple process here

    # for j in range(1, _P.shape[1].value):
    #     p3_list.append(tf.reduce_sum(tf.cumprod(p2[:, j:], axis=1), axis=1))

    # _p3 = tf.stack(p3_list, axis=1) + 1
    # p3 = tf.concat([_p3, tf.constant(1.0, shape=(_p3.shape[0].value, 1), dtype=tf.float32)], axis=1)

    # result = tf.multiply(p1, p3)
    # return tf.reshape(result, shape=P.shape.as_list())

    reshaped_P = tf.reshape(P, shape=(P.shape[0].value, -1))
    diff = reshaped_P[:, 1:] - reshaped_P[:, :-1]
    x0 = tf.slice(reshaped_P, [0, 0], [1, 1])
    diff_ = tf.concat([x0, diff], axis=1)
    result = tf.cast(tf.abs(diff_) >= 1e-7, dtype=tf.float32)
    return tf.reshape(result * weights, shape=P.shape)


def jacobian(outputs, inputs):
    jacobian_matrix = []
    M = outputs.shape[1].value
    for m in range(M):
        # We iterate over the M elements of the output vector
        grad_func = tf.gradients(outputs[0, m, 0], inputs)[0]
        jacobian_matrix.append(tf.reshape(grad_func, shape=(M, )))

    # jacobian_matrix = sess.run(jacobian_matrix)
    return ops.convert_to_tensor(jacobian_matrix, dtype=tf.float32)


def gradient_nonlinear_layer(fZ, weights=None, activation=None, reduce_sum=True):
    LOG.debug("gradient nonlinear activation {}".format(activation))
    # ignore sample
    _fZ = tf.reshape(fZ, shape=fZ.shape.as_list()[1:])
    if activation is None:
        partial_gradient = tf.keras.backend.ones(shape=_fZ.shape)
    elif activation == 'tanh':
        ### might be a bug here
        ### we need to ensure the right epoch of fZ
        partial_gradient = (1.0 + _fZ) * (1.0 - _fZ)
    elif activation == 'relu':
        _fZ = tf.reshape(fZ, shape=fZ.shape.as_list()[1:])
        partial_gradient = tf.cast(_fZ >= 1e-9, dtype=tf.float32)
    else:
        raise Exception("activation: {} not support".format(activation))

    if reduce_sum is True:
        gradient = tf.reduce_sum(partial_gradient * weights, axis=-1, keepdims=True)
    else:
        gradient = partial_gradient * weights

    return tf.reshape(gradient, shape=(fZ.shape.as_list()[:-1] + [gradient.shape[-1].value]))


def gradient_linear_layer(weights, multiples=1, expand_dims=True):
    if expand_dims is True:
        return tf.expand_dims(tf.tile(tf.transpose(weights, perm=[1, 0]), multiples=[multiples, 1]), axis=0)
    else:
        return tf.tile(tf.transpose(weights, perm=[1, 0]), multiples=[multiples, 1])


def gradient_operator_nonlinear_layers(P,
                                       fZ,
                                       operator_weights,
                                       nonlinear_weights,
                                       activation,
                                       debug=False,
                                       inputs=None,
                                       reduce_sum=True,
                                       feed_dict={}):

    if debug is True and inputs is not None:
        LOG.debug(colors.red("Only use under unittest, not for real situation"))
        J = jacobian(P, inputs)
        g1 = tf.reshape(tf.reduce_sum(J, axis=0), shape=inputs.shape)
        calc_g = gradient_operator(P, operator_weights)
        utils.init_tf_variables()
        J_result, calc_g_result = session.run([J, calc_g], feed_dict=feed_dict)
        if not np.allclose(np.diag(J_result), calc_g_result.reshape(-1)):
            LOG.error(colors.red("ERROR: gradient operator- and nonlinear- layers"))
            import ipdb; ipdb.set_trace()
    else:
        g1 = gradient_operator(P, operator_weights)

    g1 = tf.reshape(g1, shape=P.shape)
    g2 = gradient_nonlinear_layer(fZ, nonlinear_weights, activation, reduce_sum=reduce_sum)
    return g1*g2


def gradient_all_layers(P,
                        fZ,
                        operator_weights,
                        nonlinear_weights,
                        linear_weights,
                        activation,
                        debug=False,
                        inputs=None,
                        feed_dict={}):
    g1 = gradient_operator_nonlinear_layers(P, fZ,
                                            operator_weights,
                                            nonlinear_weights,
                                            activation=activation,
                                            debug=debug,
                                            inputs=inputs,
                                            reduce_sum=False,
                                            feed_dict=feed_dict)
    g2 = tf.expand_dims(tf.matmul(g1[0], linear_weights), axis=0)
    return g2


def phi(x, width=1.0):
    if x[0] > (width/2.0):
        return (x[0] - width/2.0)
    elif x[0] < (-width/2.0):
        return (x[0] + width/2.0)
    else:
        return float(0)


def confusion_matrix(y_true, y_pred, labels=['increase', 'descrease']):
    '''
    # https://en.wikipedia.org/wiki/Confusion_matrix
    confusion matrix is impolemented as following
                        | increase (truth)  | descrease (truth) |
        increase (pred) |                   |                   |
       descrease (pred) |                   |                   |
    '''

    diff1 = ((y_true[1:] - y_true[:-1]) >= 0).tolist()
    diff2 = ((y_pred[1:] - y_true[:-1]) >= 0).tolist()
    confusion = np.zeros(4, dtype=np.int32)
    for a, b in zip(diff1, diff2):
        if a is True and b is True:
            confusion[0] += 1
        elif a is True and b is False:
            confusion[2] += 1
        elif a is False and b is True:
            confusion[1] += 1
        elif a is False and b is False:
            confusion[3] += 1

    return confusion.reshape([2, 2])


class PhiCell(Layer):
    def __init__(self,
                 input_dim=1,
                 weight=1.0,
                 width=1.0,
                 hysteretic_func=Phi,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint="non_neg",
                 **kwargs):
        self.debug = kwargs.pop("debug", False)

        super(PhiCell, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self._weight = weight
        self._recurrent_weight = -1
        self._width = width
        self.units = 1
        self.state_size = [1]

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)

        self.hysteretic_func = hysteretic_func
        self.input_dim = input_dim
        self.unroll = False

    def build(self, input_shape):

        if input_shape[-1] <= 20:
            self.unroll = True

        if self.debug:
            LOG.debug("Initialize *weight* as pre-defined: {} ....".format(self._weight))
            self.kernel = tf.Variable([[self._weight]], name="weight", dtype=tf.float32)
        else:
            LOG.debug("Initialize *weight* randomly...")
            assert self.units == 1, "Phi Cell unit must be equal to 1"

            self.kernel = self.add_weight(
                "weight",
                shape=(1, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=tf.float32,
                trainable=True)

        self.built = True

    def call(self, inputs, states):
        """
        Parameters:
        ----------------
        inputs: `inputs` is a vector, the shape of inputs vector is like [1 * sequence length]
                Here, we consider the length of sequence is the same as the batch-size.
        state: `state` is randomly initialized, the shape of is [1 * 1]
        """
        self._inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)
        self._state = ops.convert_to_tensor(states[-1], dtype=tf.float32)
        LOG.debug("inputs.shape: {}".format(inputs.shape))
        LOG.debug("self._inputs.shape: {}".format(self._inputs))

        ############### IMPL from Scratch #####################
        # outputs_ = tf.multiply(self._inputs, self.kernel)
        # outputs = [self._state]
        # for i in range(outputs_.shape[-1].value):
        #     output = tf.add(Phi(tf.subtract(outputs_[0][i], outputs[-1]), self._width), outputs[-1])
        #     outputs.append(output)

        # outputs = ops.convert_to_tensor(outputs[1:], dtype=tf.float32)
        # state = outputs[-1]
        # outputs = tf.reshape(outputs, shape=self._inputs.shape)

        # LOG.debug("before reshaping state.shape: {}".format(state.shape))
        # state = tf.reshape(state, shape=(-1, 1))
        # LOG.debug("after reshaping state.shape: {}".format(state.shape))
        # return outputs, [state]

        ################ IMPL via RNN ###########################
        def steps(inputs, states):
            outputs = Phi(inputs - states[-1], self._width) + states[-1]
            return outputs, [outputs]

        self._inputs = tf.multiply(self._inputs, self.kernel)
        inputs_ = tf.reshape(self._inputs, shape=(1, self._inputs.shape[0].value*self._inputs.shape[1].value, 1))
        if isinstance(states, list) or isinstance(states, tuple):
            self._states = ops.convert_to_tensor(states[-1], dtype=tf.float32)
        else:
            self._states = ops.convert_to_tensor(states, dtype=tf.float32)

        assert self._state.shape.ndims == 2, colors.red("PhiCell states must be 2 dimensions")
        states_ = [tf.reshape(self._states, shape=self._states.shape.as_list())]
        self.unroll = True
        last_outputs_, outputs_, states_x = tf.keras.backend.rnn(steps, inputs=inputs_, initial_states=states_, unroll=self.unroll)
        return outputs_, list(states_x)


class Operator(RNN):
    def __init__(self,
                 weight=1.0,
                 width=1.0,
                 debug=False):

        cell = PhiCell(
            weight=weight,
            width=width,
            debug=debug
            )
        super(Operator, self).__init__(
            cell=cell,
            return_sequences=True,
            return_state=False,
            stateful=True,
            # unroll=False,
            unroll=True
            )

    def call(self, inputs, initial_state=None):
        LOG.debug("Operator.inputs.shape: {}".format(inputs.shape))
        output = super(Operator, self).call(inputs, initial_state=initial_state)
        assert inputs.shape.ndims == 3, colors.red("ERROR: Input from Operator must be 3 dimensions")
        shape = inputs.shape.as_list()
        return tf.reshape(output, shape=(shape[0], -1, 1))
        # return output

    @property
    def kernel(self):
        return self.cell.kernel


class MyDense(Layer):
    def __init__(self, units=1,
                 activation="tanh",
                 weight=1,
                 use_bias=True,
                 activity_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint="non_neg",
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        self._debug = kwargs.pop("debug", False)
        if '_init_kernel' in kwargs:
            self._init_kernel = kwargs.pop("_init_kernel")
        self._init_bias = kwargs.pop("_init_bias", 0)

        super(MyDense, self).__init__(**kwargs)
        self.units = units
        self._weight = weight

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        if use_bias is True:
            self.bias_initializer = initializers.get(bias_initializer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.bias_constraint = constraints.get(bias_constraint)
        self.use_bias = use_bias

    def build(self, input_shape):
        if self._debug:
            LOG.debug("init mydense kernel/bias as pre-defined")
            if hasattr(self, '_init_kernel'):
                _init_kernel = np.array([[self._init_kernel for i in range(self.units)]])
            else:
                _init_kernel = np.random.uniform(low=0.0, high=1.5, size=self.units)
            _init_kernel = _init_kernel.reshape([1, -1])
            LOG.debug(colors.yellow("kernel: {}".format(_init_kernel)))
            self.kernel = tf.Variable(_init_kernel, name="theta", dtype=tf.float32)

            if self.use_bias is True:
                # _init_bias = 0
                _init_bias = self._init_bias
                LOG.debug(colors.yellow("bias: {}".format(_init_bias)))

                self.bias = tf.Variable(_init_bias, name="bias", dtype=tf.float32)
        else:
            self.kernel = self.add_weight(
                "theta",
                shape=(1, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=tf.float32,
                trainable=True)

            if self.use_bias:
                self.bias = self.add_weight(
                    "bias",
                    shape=(1, self.units),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    dtype=tf.float32,
                    trainable=True)

        self.built = True

    def call(self, inputs):

        assert inputs.shape.ndims == 3

        # XXX: double checked. it's correct in current model. no worried
        outputs = inputs * self.kernel
        if self.use_bias:
            outputs += self.bias

        if self.activation is not None:
            outputs =  self.activation(outputs)

        # self.last_output = tf.identity(outputs)
        return outputs


class MySimpleDense(Dense):
    def __init__(self, **kwargs):
        self._debug = kwargs.pop("debug", False)
        self._init_bias = kwargs.pop("_init_bias", 0)
        if '_init_kernel' in kwargs:
            self._init_kernel = kwargs.pop("_init_kernel")

        kwargs['activation'] = None
        super(MySimpleDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.units == 1
        if self._debug is True:
            LOG.debug("init mysimpledense kernel/bias as pre-defined")
            if hasattr(self, '_init_kernel'):
                _init_kernel = np.array([self._init_kernel for i in range(input_shape[-1].value)])
            else:
                _init_kernel = np.random.uniform(low=0.0, high=1.5, size=input_shape[-1].value)
            _init_kernel = _init_kernel.reshape(-1, 1)
            LOG.debug(colors.yellow("kernel: {}".format(_init_kernel)))

            self.kernel = tf.Variable(_init_kernel, name="kernel", dtype=tf.float32)

            if self.use_bias:
                _init_bias = (self._init_bias,)
                LOG.debug(colors.yellow("bias: {}".format(_init_bias)))
                self.bias = tf.Variable(_init_bias, name="bias", dtype=tf.float32)
        else:
            super(MySimpleDense, self).build(input_shape)

        self.built = True

    def call(self, inputs):
        return super(MySimpleDense, self).call(inputs)


class Play(object):
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
                 input_dim=1):

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

    def build(self, inputs=None):
        if inputs is None and self._batch_input_shape is None:
            raise Exception("Unknown input shape")
        if inputs is not None:

            _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)

            if _inputs.shape.ndims == 1:
                length = _inputs.shape[-1].value
                if length % (self._play_input_dim * self._play_timestep) != 0:
                    LOG.error("length is: {}, input_dim: {}, play_timestep: {}".format(length,
                                                                                       self._play_input_dim,
                                                                                       self._play_timestep))
                    raise Exception("The batch size cannot be divided by the length of input sequence.")

                self.batch_size = length // (self._play_timestep * self._play_input_dim)
                self._play_batch_size = length // (self._play_timestep * self._play_input_dim)
                self._batch_input_shape = tf.TensorShape([self.batch_size, self._play_timestep, self._play_input_dim])

            else:
                raise Exception("dimension of inputs must be equal to 1")

        length = self._batch_input_shape[1].value * self._batch_input_shape[2].value
        self.batch_size = self._batch_input_shape[0].value
        assert self.batch_size == 1, colors.red("only support batch_size is 1")

        timesteps = self._batch_input_shape[1].value

        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.InputLayer(batch_size=self.batch_size,
                                                  input_shape=self._batch_input_shape[1:]))
        self.model.add(Operator(weight=getattr(self, "_weight", 1.0),
                                width=getattr(self, "_width", 1.0),
                                debug=getattr(self, "_debug", False)))
        # # TODO/NOTE: remove in future
        # self.model.add(tf.keras.layers.Reshape(target_shape=(length, 1)))

        if self._network_type == constants.NetworkType.PLAY:
            self.model.add(MyDense(self.units,
                                   activation=self.activation,
                                   use_bias=self.use_bias,
                                   debug=getattr(self, "_debug", False)))
            # self.model.add(MySimpleDense(units=1,
            #                              activation=None,
            #                              use_bias=self.use_bias,
            #                              debug=getattr(self, "_debug", False)))
            self.model.add(MySimpleDense(units=1,
                                         activation=None,
                                         use_bias=False,
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

    def predict(self, inputs, steps_per_epoch=1, verbose=0):
        inputs = ops.convert_to_tensor(inputs, tf.float32)

        if not self.built:
            self.build(inputs)

        # if inputs.shape.as_list() == self._batch_input_shape.as_list():
        #     x = inputs
        # else:
        #     x = self.reshape(inputs)
        x = inputs
        return self.model.predict(x, steps=steps_per_epoch, verbose=verbose)

    @property
    def weight(self):
        session = utils.get_session()
        if self._network_type == constants.NetworkType.OPERATOR:
            _weight = self.model._layers[1].cell.kernel
            return session.run(_weight)
        if self._network_type == constants.NetworkType.PLAY:
            phi_weight = self.model._layers[1].cell.kernel
            mydense_weights = self.model._layers[3].kernel
            dense_weights = self.model._layers[4].kernel

            weights = {}
            weights['phi_weight'] = session.run(phi_weight)
            weights['mydense_weigths'] = session.run(mydense_weights)
            weights['dense_weights'] = session.run(dense_weights)
            if self.use_bias is True:
                mydense_bias = self.model._layers[3].bias
                dense_bias = self.model._layers[4].bias
                weights['mydense_bias'] = session.run(mydense_bias)
                weights['dense_bias'] = session.run(dense_bias)

            return weights

    @property
    def number_of_layers(self):
        if not self.built:
            raise Exception("Model has not been built.")

        return len(self.model._layers)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path, by_name=False):
        self.model.load_weights(path, by_name=by_name)

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


class MyModel(object):
    def __init__(self, nb_plays=1,
                 units=1,
                 batch_size=1,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 activation='tanh',
                 optimizer='adam',
                 timestep=1,
                 input_dim=1,
                 diff_weights=False,
                 network_type=constants.NetworkType.PLAY,
                 learning_rate=0.001
                 ):
        # fix random seed to 123
        seed = 123
        np.random.seed(seed)
        LOG.debug(colors.red("Make sure you are using the right random seed. currently seed is {}".format(seed)))

        self.plays = []
        self._nb_plays = nb_plays
        self._activation = activation
        self._input_dim = input_dim
        _weight = 1.0
        _width = 0.1
        width = 1
        for nb_play in range(nb_plays):
            if diff_weights is True:
                # weight = 0.5 / (_width * (1 + nb_play)) # width range from (0.1, ... 0.1 * nb_plays)
                weight = 0.5 / (_width * (1 + nb_play)) # width range from (0.1, ... 0.1 * nb_plays)
            else:
                weight = 1.0

            LOG.debug("MyModel geneartes {} with Weight: {}".format(colors.red("Play #{}".format(nb_play+1)), weight))

            play = Play(units=units,
                        batch_size=batch_size,
                        weight=weight,
                        width=width,
                        debug=debug,
                        activation=activation,
                        loss=None,
                        optimizer=None,
                        network_type=network_type,
                        name="play-{}".format(nb_play),
                        timestep=timestep,
                        input_dim=input_dim)
            assert play._need_compile == False, colors.red("Play inside MyModel mustn't be compiled")
            self.plays.append(play)

        # self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

    def fit(self,
            inputs,
            outputs,
            epochs=100,
            verbose=0,
            steps_per_epoch=1,
            loss_file_name="./tmp/mymodel_loss_history.csv",
            learning_rate=0.001,
            decay=0.):

        writer = utils.get_tf_summary_writer("./log/mse")

        inputs = ops.convert_to_tensor(inputs, tf.float32)
        __mu__ = (outputs[1:] - outputs[:-1]).mean()
        __sigma__ = (outputs[1:] - outputs[:-1]).std()
        outputs = ops.convert_to_tensor(outputs, tf.float32)

        for play in self.plays:
            if not play.built:
                play.build(inputs)

        _xs = []
        _ys = []
        for play in self.plays:
            _x, _y = play.reshape(inputs, outputs)
            _xs.append(_x)
            _ys.append(_y)

        params_list = []
        model_outputs = []
        feed_inputs = []
        feed_targets = []

        for idx, play in enumerate(self.plays):
            feed_inputs.append(play.input)
            model_outputs.append(play.output)

            shape = tf.keras.backend.int_shape(play.output)
            name = 'play{}_target'.format(idx)
            target = tf.keras.backend.placeholder(
                ndim=len(shape),
                name=name,
                dtype=tf.float32)
            feed_targets.append(target)

            # update_inputs += play.model.get_updates_for(inputs)
            params_list += play.trainable_weights

        if self._nb_plays > 1:
            y_pred = tf.keras.layers.Average()(model_outputs)
        else:
            y_pred = model_outputs[0]

        loss = tf.keras.backend.mean(tf.math.square(y_pred - _y))
        mu = tf.keras.backend.mean(y_pred[:, 1:, :] - y_pred[:, :-1, :])
        sigma = tf.keras.backend.std(y_pred[:, 1:, :] - y_pred[:, :-1, :])
        # decay: decay learning rate to half every 100 steps
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay)
        with tf.name_scope('training'):
            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=params_list,
                                                     loss=loss)
            # updates += update_inputs

            training_inputs = feed_inputs + feed_targets
            train_function = tf.keras.backend.function(training_inputs,
                                                       [loss, mu, sigma],
                                                       updates=updates)

        # _x = [x for _ in range(self._nb_plays)]
        # _y = [y for _ in range(self._nb_plays)]
        ins = _xs + _ys

        self.cost_history = []

        path = "/".join(loss_file_name.split("/")[:-1])
        writer.add_graph(tf.get_default_graph())
        loss_summary = tf.summary.scalar("loss", loss)

        for i in range(epochs):
            for j in range(steps_per_epoch):
                cost, predicted_mu, predicted_sigma = train_function(ins)
            self.cost_history.append([i, cost])
            LOG.debug("Epoch: {}, Loss: {}, predicted_mu: {}, predicted_sigma: {}, truth_mu: {}, truth_sigma: {}".format(i,
                                                                                                                         cost,
                                                                                                                         predicted_mu,
                                                                                                                         predicted_sigma,
                                                                                                                         __mu__,
                                                                                                                         __sigma__))

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict(self, inputs, individual=False):
        inputs = ops.convert_to_tensor(inputs, tf.float32)

        for play in self.plays:
            if not play.built:
                play.build(inputs)

        x = self.plays[0].reshape(inputs)
        outputs = []

        for play in self.plays:
            start = time.time()
            outputs.append(play.predict(x))
            end = time.time()
            LOG.debug("play {} cost time {} s".format(play._name, end-start))

        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        if individual is True:
            outputs_ = outputs_.reshape(len(self.plays), -1).T
            # sanity checking
            for i in range(len(self.plays)):
                if np.all(outputs_[i, :] == outputs[i]) is False:
                    raise
            return prediction, outputs_
        return prediction

    @property
    def weights(self):
        for play in self.plays:
            LOG.debug("{}, number of layer is: {}".format(play._name, play.number_of_layers))
            LOG.debug("{}, weight: {}".format(play._name, play.weight))

    def compile(self, inputs, **kwargs):
        unittest = kwargs.pop('unittest', False)
        mu = kwargs.pop('mu', None)
        sigma = kwargs.pop('sigma', None)
        LOG.debug("Compile with mu: {}, sigma: {}".format(mu, sigma))

        _inputs = inputs
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        for play in self.plays:
            if not play.built:
                play.build(inputs)

        self.params_list = []
        self.feed_inputs = []
        self.model_outputs = []

        self.feed_mu = tf.constant(mu, name='input-mu', dtype=tf.float32)
        self.feed_sigma= tf.constant(sigma, name='input-sigma', dtype=tf.float32)

        self.mu_placeholder = tf.keras.backend.placeholder(ndim=0,
                                                           name='target-mu',
                                                           dtype=tf.float32)
        self.sigma_placeholder = tf.keras.backend.placeholder(ndim=0,
                                                              name='target-sigma',
                                                              dtype=tf.float32)

        # TODO: not feed with self._x, can't be a bug HERE

        # self.feed_targets = [self.feed_mu, self.feed_sigma]
        # self.feed_targets = [self.mu_placeholder, self.sigma_placeholder]
        self.feed_targets = []
        self.update_inputs = []
        for play in self.plays:
            self.feed_inputs.append(play.input)
            self.model_outputs.append(play.output)
            # TODO: figure out the function of get_updates_for
            self.update_inputs += play.model.get_updates_for(play.input)
            self.params_list += play.trainable_weights

        self._x = [play.reshape(inputs) for play in self.plays]
        self._y = [self.feed_mu, self.feed_sigma]
        self._x_feed_dict = {self.feed_inputs[k].name : _inputs.reshape(1, -1, self._input_dim) for k in range(self._nb_plays)}

        ##################### Average outputs #############################
        if self._nb_plays > 1:
            self.y_pred = tf.keras.layers.Average()(self.model_outputs)
        else:
            self.y_pred = self.model_outputs[0]

        with tf.name_scope('training'):
            diff = self.y_pred[:, 1:, :] - self.y_pred[:, :-1, :]

            if unittest is False:
                ###################### Calculate J by hand ###############################
                J_list = [gradient_all_layers(play.operator_layer.output,
                                              play.nonlinear_layer.output,
                                              play.operator_layer.kernel,
                                              play.nonlinear_layer.kernel,
                                              play.linear_layer.kernel,
                                              activation=self._activation) for play in self.plays]
            else:
                ###################### Calculate J by hand ###############################
                J_list = [gradient_all_layers(play.operator_layer.output,
                                              play.nonlinear_layer.output,
                                              play.operator_layer.kernel,
                                              play.nonlinear_layer.kernel,
                                              play.linear_layer.kernel,
                                              activation=self._activation,
                                              debug=True,
                                              inputs=self.feed_inputs[i],
                                              feed_dict=self._x_feed_dict) for i, play in enumerate(self.plays)]
                self.J_list_by_hand = J_list
                ####################### Calculate J by Tensorflow ###############################
                y_pred = tf.reshape(self.y_pred, shape=self.feed_inputs[0].shape)
                J_by_tf = tf.keras.backend.gradients(y_pred, self.feed_inputs)
                J_by_tf = [tf.reshape(J_by_tf[i], shape=(1, -1, 1)) for i in range(self._nb_plays)]
                self.J_by_tf = tf.reduce_mean(tf.concat(J_by_tf, axis=-1), axis=-1, keepdims=True)

                model_outputs = [tf.reshape(self.model_outputs[i], shape=self.feed_inputs[i].shape) for i in range(self._nb_plays)]
                J_list_by_tf = tf.keras.backend.gradients(model_outputs, self.feed_inputs)
                self.J_list_by_tf = [tf.reshape(J_list_by_tf[i], shape=(1, -1, 1)) for i in range(self._nb_plays)]


            # by tf
            # model_outputs = [tf.reshape(self.model_outputs[i], shape=self.feed_inputs[i].shape) for i in range(self._nb_plays)]
            # J_list_by_tf = tf.keras.backend.gradients(model_outputs, self.feed_inputs)
            # self.J_list_by_tf = [tf.reshape(J_list_by_tf[i], shape=(1, -1, 1)) for i in range(self._nb_plays)]
            # self.J_by_tf = tf.reduce_mean(tf.concat(self.J_list_by_tf, axis=-1), axis=-1, keepdims=True) / self._nb_plays
            normalized_J_by_tf = tf.clip_by_value(tf.abs(self.J_by_tf), clip_value_min=1e-9, clip_value_max=1e9)

            # import ipdb; ipdb.set_trace()
            # by hand
            # (T * 1)
            self.J_by_hand = tf.reduce_mean(tf.concat(J_list, axis=-1), axis=-1, keepdims=True) / self._nb_plays
            normalized_J_by_hand = tf.clip_by_value(tf.abs(self.J_by_hand), clip_value_min=1e-9, clip_value_max=1e9)

            ################################################################################
            # TODO: support derivation for p0
            # TODO: make loss customize from outside
            self.curr_mu = tf.keras.backend.mean(diff)
            self.curr_sigma = tf.keras.backend.std(diff)
            ## learn mu/sigma/weights
            # _loss = tf.keras.backend.square((diff - self.curr_mu)/self.curr_sigma) / 2 - tf.keras.backend.log(normalized_J[:, 1:, :])
            ## fix mu and learn sigma/weights
            # _loss = tf.keras.backend.square((diff - mu)/self.curr_sigma) / 2 - tf.keras.backend.log(normalized_J[:, 1:, :])
            ## fix mu/sigma and learn weights
            mu = -0.0032573
            # sigma = 6.9357114
            sigma = 8.4
            self.loss_a = tf.keras.backend.square((diff - mu)/sigma) / 2
            self.loss_b = - tf.keras.backend.log(normalized_J_by_hand[:, 1:, :])
            self.loss_c = - tf.keras.backend.log(normalized_J_by_tf[:, 1:, :])
            # _loss = tf.keras.backend.square((diff - mu)/sigma) / 2 - tf.keras.backend.log(normalized_J[:, 1:, :])
            # _loss = self.loss_a + self.loss_b
            _loss = self.loss_a + self.loss_c
            self.loss_by_tf = tf.keras.backend.mean(_loss)
            self.loss_by_hand = tf.keras.backend.mean(self.loss_a + self.loss_b)
            self.loss = self.loss_by_tf

    def fit2(self,
             inputs,
             mu,
             sigma,
             outputs=None,
             epochs=100,
             verbose=0,
             steps_per_epoch=1,
             loss_file_name="./tmp/mymodel_loss_history.csv",
             learning_rate=0.001,
             decay=0.):

        writer = utils.get_tf_summary_writer("./log/mle")

        if outputs is not None:
            # glob ground-truth mu and sigma of outputs
            __mu__ = (outputs[1:] - outputs[:-1]).mean()
            __sigma__ = (outputs[1:] - outputs[:-1]).std()
            outputs = ops.convert_to_tensor(outputs, tf.float32)

        self.compile(inputs, mu=mu, sigma=sigma, unittest=True)
        mse_loss1 = tf.keras.backend.mean(tf.square(self.y_pred - tf.reshape(outputs, shape=self.y_pred.shape)))
        mse_loss2 = tf.keras.backend.mean(tf.square(self.y_pred + tf.reshape(outputs, shape=self.y_pred.shape)))
        diff = self.y_pred[:, 1:, :] - self.y_pred[:, :-1, :]

        with tf.name_scope(self.optimizer.__class__.__name__):
            updates = self.optimizer.get_updates(params=self.params_list,
                                                 loss=self.loss)

            # updates += self.update_inputs
            training_inputs = self.feed_inputs + self.feed_targets
            # training_inputs = self.feed_inputs
            train_function = tf.keras.backend.function(training_inputs,
                                                       [self.loss, mse_loss1, mse_loss2, diff, self.curr_sigma, self.curr_mu, self.loss_by_hand, self.loss_by_tf],
                                                       # [self.loss],
                                                       updates=updates)

        # ins = self._x + self._y
        ins = self._x
        utils.init_tf_variables()

        writer.add_graph(tf.get_default_graph())
        self.cost_history = []
        cost = np.inf
        patience_list = []
        prev_cost = np.inf

        utils.init_tf_variables()
        for i in range(epochs):
            for j in range(steps_per_epoch):
                cost, mse_cost1, mse_cost2, diff_res, sigma_res, mu_res, loss_by_hand, loss_by_tf = train_function(ins)
                if prev_cost <= cost:
                    patience_list.append(cost)
                else:
                    prev_cost = cost
                    patience_list = []
            # LOG.debug("Epoch: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, placeholder_sigma: {:.7f}, placeholder_mu: {:.7f}".format(i,
            #                                                                                                                                                                                                     float(cost),
            #                                                                                                                                                                                                     float(mse_cost1),
            #                                                                                                                                                                                                     float(mse_cost2),
            #                                                                                                                                                                                                     float(diff_res.mean()),
            #                                                                                                                                                                                                     float(diff_res.std()),
            #                                                                                                                                                                                                     float(__mu__),
            #                                                                                                                                                                                                     float(__sigma__),
            #                                                                                                                                                                                                     float(sigma_res),
            #                                                                                                                                                                                                     float(mu_res)))
            LOG.debug("Epoch: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}".format(i,
                                                                                                                                                                                                       float(cost),
                                                                                                                                                                                                       float(mse_cost1),
                                                                                                                                                                                                       float(mse_cost2),
                                                                                                                                                                                                       float(diff_res.mean()),
                                                                                                                                                                                                       float(diff_res.std()),
                                                                                                                                                                                                       float(__mu__),
                                                                                                                                                                                                       float(__sigma__),
                                                                                                                                                                                                       float(loss_by_hand),
                                                                                                                                                                                                       float(loss_by_tf)))


            # LOG.debug("Epoch: {}, max(loss_a): {:.7f}, min(loss_a): {:.7f}, max(loss_b): {:.7f}, min(loss_b): {:.7f}".format(i,
            #                                                                                                                  float(np.max(loss_a)),
            #                                                                                                                  float(np.min(loss_a)),
            #                                                                                                                  float(np.max(loss_b)),
            #                                                                                                                  float(np.min(loss_b))))
            # LOG.debug("Epoch: {}, loss_a: {}".format(i,
            #                                          loss_a.reshape(-1)))

            # LOG.debug("Epoch: {}, loss_b: {}".format(i,
            #                                          loss_b.reshape(-1)))

            # LOG.debug("Epoch: {}, J_by_hand: {}".format(i,
            #                                             J_by_hand.reshape(-1)))
            LOG.debug("================================================================================")


            self.cost_history.append([i, cost])
            # if len(patience_list) >= 50:
            #     LOG.debug(colors.yellow("Lost patience...."))
            #     break

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict2(self, inputs):
        _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)
        for play in self.plays:
            self._need_compile = False
            if not play.built:
                play.build(_inputs)

        outputs = []
        for play in self.plays:
            x = play.reshape(inputs)
            outputs.append(play.predict(x))
        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)

        # keep prices and random walk in the same direction in our assumption
        diff_pred = prediction[1:] - prediction[:-1]
        diff_input = inputs[1:] - inputs[:-1]
        prediction = utils.slide_window_average(prediction, window_size=1)

        mean = diff_pred.mean()
        std = diff_pred.std()
        LOG.debug("mean: {}, std: {}".format(mean, std))
        return prediction, float(mean), float(std)

    def trend(self, prices, B, mu, sigma,
              start_pos=1000, end_pos=1100,
              delta=0.001, max_iteration=10000):
        # end_pos = 1020
        assert start_pos < end_pos, colors.red("start_pos must be less than end_pos")
        assert len(prices.shape) == 1, colors.red("Prices should be a vector")
        if not hasattr(self, '_batch_input_shape'):
            if hasattr(self.plays[0], '_batch_input_shape'):
                input_dim = self._batch_input_shape.as_list()[-1]
            else:
                raise Exception(colors.red("Not found batch_input_shape"))
        elif isinstance(self._batch_input_shape, (list, tuple)):
            input_dim = self._batch_input_shape[-1]
        else:
            raise Exception(colors.red("Unknown error occurs in trend"))

        timestep = prices.shape[0] // input_dim
        shape = [1, timestep, input_dim]
        _ppp = ops.convert_to_tensor(prices, dtype=tf.float32)
        _ppp = tf.reshape(_ppp, shape=shape)

        original_prediction = self.predict(prices)
        mu = (original_prediction[1:start_pos] - original_prediction[:start_pos-1]).mean()
        sigma = (original_prediction[1:start_pos] - original_prediction[:start_pos-1]).std()

        weights_ = [[], [], [], [], []]
        operator_outputs_ = []

        for play in self.plays:
            weights_[0].append(play.operator_layer.kernel)
            weights_[1].append(play.nonlinear_layer.kernel)
            weights_[2].append(play.nonlinear_layer.bias)
            weights_[3].append(play.linear_layer.kernel)
            weights_[4].append(play.linear_layer.bias)
            operator_outputs_.append(play.operator_layer(_ppp))

        weights, operator_outputs = sess.run([weights_, operator_outputs_])

        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = weights[i][j].reshape(-1)

        # if start_pos == 0:
        #     individual_p_list = [[0] for operator_output in operator_outputs]
        # else:
        #     individual_p_list = [[operator_output.reshape(-1)[start_pos-1]] for operator_output in operator_outputs]
        # individual_p_list = [operator_output.reshape(-1)[start_pos-1:end_pos-1] for operator_output in operator_outputs]
        individual_p_list = [[operator_output.reshape(-1)[start_pos-1]] for operator_output in operator_outputs]
        # import ipdb; ipdb.set_trace()
        outputs = []

        def do_guess_helper(k, step=1, direction=1, guess_flag=True, input_p=None, base_price=0):
            predict_noise_list = []
            # guess = prices[k]
            # if input_p is not None:
            #     guess = input_p

            # for i in range(self._nb_plays):
            #     p = phi(weights[0][i] * guess - individual_p_list[i][k-start_pos]) + individual_p_list[i][k-start_pos]
            #     pp = weights[1][i] * p + weights[2][i]
            #     if self._activation is None:
            #         pass
            #     elif self._activation == 'tanh':
            #         pp = np.tanh(pp)
            #     elif self._activation == 'relu':
            #         pp = np.relu(pp)

            #     ppp = (weights[3][i] * pp).sum() + weights[4][i]
            #     predict_noise_list.append(ppp[0])

            # predict_noise = sum(predict_noise_list) / self._nb_plays
            # if not np.allclose(predict_noise, original_prediction[k], atol=1e-5):
            #     LOG.debug(colors.red("ERROR: predict_noise: {}, base_noise: {}".format(
            #         predict_noise, original_prediction[k]
            #     )))

            #     import ipdb; ipdb.set_trace()

            # if guess_flag is False:
            #     return guess,  predict_noise


            ############################################################################################################
            # predict_noise_list = []

            # guess = prices[k-1] + direction * step * delta
            guess = base_price + direction * step * delta

            for i in range(self._nb_plays):
                p = phi(weights[0][i] * guess - individual_p_list[i][k-start_pos]) + individual_p_list[i][k-start_pos]
                pp = weights[1][i] * p + weights[2][i]
                if self._activation is None:
                    pass
                elif self._activation == 'tanh':
                    pp = np.tanh(pp)
                elif self._activation == 'relu':
                    pp = np.relu(pp)

                ppp = (weights[3][i] * pp).sum() + weights[4][i]
                predict_noise_list.append(ppp[0])

            predict_noise = sum(predict_noise_list) / self._nb_plays

            return guess, predict_noise


        def do_guess_seq(start, seq=1):
            guess_price_seq = [prices[start-1]]
            predict_noise_seq = [original_prediction[start-1]]
            interval = 0
            while interval < seq:
                k = start + interval
                bk = np.random.normal(loc=mu, scale=sigma) + predict_noise_seq[-1]

                if bk > predict_noise_seq[-1]:
                    direction = -1
                elif bk < predict_noise_seq[-1]:
                    direction = 1
                else:
                    direction = 0

                step = 1
                guess, guess_noise = do_guess_helper(k, step, direction, base_price=guess_price_seq[-1])
                curr_diff = guess_noise - bk
                prev_diff = None
                good_guess = True
                # import ipdb; ipdb.set_trace()
                while True:
                    prev_diff = curr_diff

                    step += 1
                    guess, guess_noise = do_guess_helper(k, step, direction, base_price=guess_price_seq[-1])
                    curr_diff = guess_noise - bk

                    if curr_diff * prev_diff < 0:
                        LOG.debug(colors.yellow("step: {}, true_price: {:.5f}, guess price: {:.5f}, guess noise: {:.5f}, generated noise: {:.5f}, true noise: {:.5f}, prev true noise: {:.5f}, curr_diff: {:.5f}, prev_diff: {:.5f}, direction: {}, delta: {}".format(
                            step,
                            float(prices[k]),
                            float(guess),
                            float(guess_noise),
                            float(bk),
                            float(original_prediction[k]),
                            float(original_prediction[k-1]),
                            float(curr_diff),
                            float(prev_diff),
                            direction,
                            delta)))
                        break

                    # LOG.debug("step: {}, true_price: {:.5f}, guess price: {:.5f}, guess noise: {:.5f}, generated noise: {:.5f}, true noise: {:.5f}, curr_diff: {:.5f}, prev_diff: {:.5f}, direction: {}, delta: {}".format(
                    #     step,
                    #     float(prices[k]),
                    #     float(guess),
                    #     float(guess_noise),
                    #     float(bk),
                    #     float(original_prediction[k]),
                    #     float(curr_diff),
                    #     float(prev_diff),
                    #     direction,
                    #     delta))

                    if step >= 300:
                        LOG.warn(colors.red("Not a good guess"))
                        good_guess = False
                        break

                if good_guess is False:
                    continue

                guess_price_seq.append(guess)
                predict_noise_seq.append(guess_noise)
                for j in range(self._nb_plays):
                    p = phi(weights[0][j] * guess - individual_p_list[j][-1]) + individual_p_list[j][-1]
                    individual_p_list[j].append(p)

                interval += 1

            # import ipdb; ipdb.set_trace()
            return guess_price_seq[1:]


        guess_prices_list = [[] for _ in range(end_pos-start_pos)]
        delta = 0.001
        np.random.seed(int(time.time()))
        # import ipdb; ipdb.set_trace()

        # TODO: automatically adjust the delta to search the results

        # def repeat(k, iterations=1):
        #     i = 0
        #     while i < iterations:
        #         # import ipdb;ipdb.set_trace()

        #         bk = np.random.normal(loc=mu, scale=sigma) + original_prediction[k-1]
        #         if bk > original_prediction[k-1]:
        #             direction = -1
        #         elif bk < original_prediction[k-1]:
        #             direction = 1
        #         else:
        #             direction = 0

        #         book_prev_diff_list = []
        #         guess, guess_noise = do_guess_helper(k, 1, direction)
        #         step = 1
        #         curr_diff = guess_noise - bk
        #         prev_diff = None
        #         LOG.debug(colors.cyan("============================= Iteration #{} ======================================".format(i+1)))
        #         good_guess = True

        #         while True:
        #             prev_diff = curr_diff
        #             book_prev_diff_list.append(prev_diff)

        #             step += 1
        #             guess, guess_noise = do_guess_helper(k, step, direction, guess_flag=True)
        #             curr_diff = guess_noise - bk

        #             if curr_diff * prev_diff < 0:
        #                 LOG.debug(colors.yellow("step: {}, true_price: {:.5f}, guess price: {:.5f}, guess noise: {:.5f}, generated noise: {:.5f}, true noise: {:.5f}, prev true noise: {:.5f}, curr_diff: {:.5f}, prev_diff: {:.5f}, direction: {}, delta: {}".format(
        #                     step,
        #                     float(prices[k]),
        #                     float(guess),
        #                     float(guess_noise),
        #                     float(bk),
        #                     float(original_prediction[k]),
        #                     float(original_prediction[k-1]),
        #                     float(curr_diff),
        #                     float(prev_diff),
        #                     direction,
        #                     delta)))
        #                 guess_prices_list[k-start_pos].append(guess)
        #                 break

        #             # LOG.debug("step: {}, true_price: {:.5f}, guess price: {:.5f}, guess noise: {:.5f}, generated noise: {:.5f}, true noise: {:.5f}, curr_diff: {:.5f}, prev_diff: {:.5f}, direction: {}, delta: {}".format(
        #             #     step,
        #             #     float(prices[k]),
        #             #     float(guess),
        #             #     float(guess_noise),
        #             #     float(bk),
        #             #     float(original_prediction[k]),
        #             #     float(curr_diff),
        #             #     float(prev_diff),
        #             #     direction,
        #             #     delta))

        #             # if step % 10 == 0:
        #             #     import ipdb; ipdb.set_trace()
        #             if step >= 300:
        #                 LOG.warn(colors.red("Not a good guess"))
        #                 good_guess = False
        #                 break

        #         if good_guess is True:
        #             i += 1

        #     avg_guess = sum(guess_prices_list[k-start_pos]) / iterations
        #     for j in range(self._nb_plays):
        #         # p = phi(weights[0][j] * avg_guess - individual_p_list[j][-1]) + individual_p_list[j][-1]
        #         # individual_p_list[j].append(p)
        #         individual_p_list[j].append(operator_outputs[j].reshape(-1)[k])
        #     return avg_guess

        def repeat(k, seq=1, iteration=1):
            guess_price_seq_stack = []
            for i in range(iteration):
                guess_price_seq = do_guess_seq(k, seq=seq)
                guess_price_seq_stack.append(guess_price_seq)
                guess_prices_list[k-start_pos+seq-1].append(guess_price_seq[-1])

            guess_price_seq_stack_ = np.array(guess_price_seq_stack)
            return guess_price_seq_stack_.mean(axis=0)[-1]

        guess_prices = []

        k = start_pos
        seq = 1
        iteration = 100
        for s in range(seq-1):
            guess_prices.append(0)
            guess_prices_list[k-start_pos+s] = [0]*iteration

        while True:
            avg_guess = repeat(k, seq=seq, iteration=iteration)
            guess_prices.append(avg_guess)
            # import ipdb; ipdb.set_trace()
            LOG.debug(colors.red("================ Guess k: {} successfully, predict price: {:.5f}, grouth-truth price: {:.5f} prev gt price: {:.5f} =====================".format(k+seq-1,
                                                                                                                                                                                   float(avg_guess),
                                                                                                                                                                                   float(prices[k]),
                                                                                                                                                                                   float(prices[k-1]))))

            k += 1
            if (k + seq - 1)>= end_pos:
                break



        LOG.debug("Verifing...")
        outputs = np.array(outputs).reshape(-1)
        guess_prices = np.array(guess_prices).reshape(-1)

        loss1 =  ((guess_prices - prices[start_pos:end_pos]) ** 2)
        loss2 = np.abs(guess_prices - prices[start_pos:end_pos])
        loss3 = (prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1]) ** 2
        loss4 = np.abs(prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1])

        LOG.debug("root sum square loss1: {}".format((loss1.sum())**(0.5)))
        LOG.debug("root sum square loss2: {}".format((loss3.sum())**(0.5)))
        LOG.debug("total abs loss1: {}".format((loss2.sum())))
        LOG.debug("total abs loss2: {}".format((loss4.sum())))

        guess_prices_list = np.array(guess_prices_list)

        # LOG.debug()
        # TODO: calculate confusion matrix HERE
        return guess_prices, guess_prices_list

    def save_weights(self, fname):
        suffix = fname.split(".")[-1]
        for play in self.plays:
            path = "{}/{}plays/{}.{}".format(fname[:-3], len(self.plays), play._name, suffix)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()
            LOG.debug(colors.cyan("Saving {}'s Weights to {}".format(play._name, path)))
            play.save_weights(path)

        LOG.debug(colors.cyan("Writing input shape into disk..."))
        with open("{}/{}plays/input_shape.txt".format(fname[:-3], len(self.plays)), "w") as f:
            f.write(":".join(map(str, self.plays[0]._batch_input_shape.as_list())))

    def load_weights(self, fname, extra={}):
        LOG.debug(colors.cyan("Trying to Load Weights first..."))
        suffix = fname.split(".")[-1]
        dirname = "{}/{}plays".format(fname[:-3], len(self.plays))
        if not os.path.isdir(dirname):
            LOG.debug(colors.red("Fail to Load Weights."))
            return
        LOG.debug(colors.red("Found trained Weights. Loading..."))
        with open("{}/input_shape.txt".format(dirname), "r") as f:
            line = f.read()

        shape = list(map(int, line.split(":")))
        if 'shape' in extra:
            shape = extra['shape']

        self._batch_input_shape = shape

        for play in self.plays:
            if not play.built:
                play._batch_input_shape = tf.TensorShape(shape)
                play._preload_weights = True
                play.build()
            path = "{}/{}.{}".format(dirname, play._name, suffix)
            play.load_weights(path)
            LOG.debug(colors.red("Set Weights for {}".format(play._name)))
