import os
import tensorflow as tf
# tf.enable_eager_execution()
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
import numpy as np

import utils
import colors
import constants
import trading_data as tdata
import log as logging
import pool
import pickle


LOG = logging.getLogger(__name__)

# tf.keras.backend.set_session(
#     tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:1234"))

sess = utils.get_session()
session = utils.get_session()
SESS = utils.get_session()
SESSION = utils.get_session()


tf.keras.backend.set_epsilon(1e-7)


def Phi(x, width=1.0):
    '''
    Phi(x) = x         , if x > 0
           = x + width , if x < - width
           = 0         , otherwise
    '''
    assert x.shape[0].value == 1 and x.shape[1].value == 1, "x must be a scalar"

    ZEROS = tf.zeros(x.shape, dtype=tf.float32, name='zeros')
    _width = tf.constant([[width/2.0]], dtype=tf.float32)
    r1 = tf.cond(tf.reduce_all(tf.less(x, -_width)), lambda: x - _width, lambda: ZEROS)
    r2 = tf.cond(tf.reduce_all(tf.greater(x, _width)), lambda: x + _width, lambda: r1)
    return r2


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
        # self.last_kernel = self.add_weight(
        #     "last_weight",
        #     shape=(1, 1),
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     dtype=tf.float32,
        #     trainable=False)

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

        # # NOTE: unroll method, can we use RNN method ?
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
            unroll=False,
            )

    def call(self, inputs, initial_state=None):
        LOG.debug("Operator.inputs.shape: {}".format(inputs.shape))
        output = super(Operator, self).call(inputs, initial_state=initial_state)
        assert inputs.shape.ndims == 3, colors.red("ERROR: Input from Operator must be 3 dimensions")
        shape = inputs.shape.as_list()

        return tf.reshape(output, shape=(shape[0], -1, 1))

    @property
    def kernel(self):
        return self.cell.kernel


    # @property
    # def last_kernel(self):
    #     return self.cell.last_kernel


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
        # self.last_kernel = self.add_weight(
        #     "last_theta",
        #     shape=(1, self.units),
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     dtype=tf.float32,
        #     trainable=False)

        if self._debug:
            LOG.debug("init mydense kernel/bias as pre-defined")
            # _init_kernel = np.array([[1/2 * (i + 1) for i in range(self.units)]])
            # _init_kernel = np.array([[1 for i in range(self.units)]])
            _init_kernel = np.random.uniform(low=0.0, high=1.5, size=self.units)
            _init_kernel = _init_kernel.reshape([1, -1])
            LOG.debug(colors.yellow("kernel: {}".format(_init_kernel)))
            self.kernel = tf.Variable(_init_kernel, name="theta", dtype=tf.float32)
            # self._trainable_weights.append(self.kernel)

            if self.use_bias is True:
                _init_bias = 0
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

        kwargs['activation'] = None
        super(MySimpleDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.units == 1
        # self.last_kernel = self.add_weight(
        #     "last_kernel",
        #     shape=(input_shape[-1].value, 1),
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     dtype=tf.float32,
        #     trainable=False)

        if self._debug is True:
            LOG.debug("init mysimpledense kernel/bias as pre-defined")
            # _init_kernel = np.array([1 * (i + 1) for i in range(input_shape[-1].value)])
            # _init_kernel = np.array([1 for i in range(input_shape[-1].value)])
            _init_kernel = np.random.uniform(low=0.0, high=1.5, size=input_shape[-1].value)
            _init_kernel = _init_kernel.reshape(-1, 1)
            LOG.debug(colors.yellow("kernel: {}".format(_init_kernel)))

            self.kernel = tf.Variable(_init_kernel, name="kernel", dtype=tf.float32)

            if self.use_bias:
                _init_bias = (0,)
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

        # if debug:
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
                # timesteps = length // self.batch_size
                # if timesteps * self.batch_size != length:
                #     raise Exception("The batch size cannot be divided by the length of input sequence.")
                # self._batch_input_shape = tf.TensorShape([1, timesteps, self.batch_size])

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
        # TODO/NOTE: remove in future
        self.model.add(tf.keras.layers.Reshape(target_shape=(length, 1)))

        if self._network_type == constants.NetworkType.PLAY:
            self.model.add(MyDense(self.units,
                                   activation=self.activation,
                                   use_bias=self.use_bias,
                                   debug=getattr(self, "_debug", False)))
            self.model.add(MySimpleDense(units=1,
                                         activation=None,
                                         use_bias=self.use_bias,
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

        if not getattr(self, "_preload_weights", False):
            utils.init_tf_variables()
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

    def predict(self, inputs, steps_per_epoch=1):
        inputs = ops.convert_to_tensor(inputs, tf.float32)

        if not self.built:
            self.build(inputs)

        x = self.reshape(inputs)

        return self.model.predict(x, steps=steps_per_epoch)

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

    def fit2(self, inputs, mean, sigma, epochs, verbose=0, steps_per_epoch=1, loss_file_name="./tmp/tmp.csv"):

        _inputs = ops.convert_to_tensor(inputs, tf.float32)
        if not self.built:
            self._need_compile = False
            self.build(_inputs)

        self.model.optimizer = optimizers.get(self.optimizer)

        x = self.reshape(_inputs)

        self._model_input = self.model.layers[0].input
        self._model_output = self.model.layers[-1].output

        self.mean = tf.Variable(mean, name="mean", dtype=tf.float32)
        self.std = tf.Variable(sigma, name="sigma", dtype=tf.float32)

        feed_inputs = [self._model_input]
        target_mean = tf.keras.backend.placeholder(
            ndim=0,
            name="mean_target",
            dtype=tf.float32
        )
        target_std = tf.keras.backend.placeholder(
            ndim=0,
            name="sigma_target",
            dtype=tf.float32
        )
        feed_targets = [target_mean, target_std]

        # import ipdb; ipdb.set_trace()
        with tf.name_scope('training'):
            J = tf.keras.backend.gradients(self._model_output, self._model_input)
            detJ = tf.reshape(tf.keras.backend.abs(J[0]), shape=self._model_output.shape)
            # avoid zeros
            detJ = tf.keras.backend.clip(detJ, min_value=1e-5, max_value=1e9)

            diff = self._model_output[:, 1:, :] - self._model_output[:, :-1, :]
            # _loss = (tf.keras.backend.square((diff - self.mean) / self.std) + tf.keras.backend.log(self.std*self.std)) + tf.keras.backend.log(detJ[:, 1:, :])
            _loss = tf.keras.backend.square((diff-self.mean)/self.std)/2.0 - tf.keras.backend.log(detJ[:, 1:, :])
            loss = tf.keras.backend.mean(_loss)

            params = self.model.trainable_weights

            with tf.name_scope(self.model.optimizer.__class__.__name__):
                updates = self.model.optimizer.get_updates(params=params,
                                                           loss=loss)

            updates += self.model.get_updates_for(self._model_input)

            inputs = feed_inputs + feed_targets
            self.train_function = tf.keras.backend.function(inputs,
                                                            [loss, self._model_output, detJ],
                                                            # [loss],
                                                            updates=updates)
        self._max_retry = 10
        self.retry = False
        self.cost_history = []

        def fit2_loop(ins):
            self.cost_history = []
            self.retry = False
            i = 0
            prev_cost = np.inf
            patient_list = []
            start_flag = True
            cost = np.inf
            while i < epochs:
                i += 1
                for j in range(steps_per_epoch):  # bugs when steps_per_epoch == 1
                    prev_cost = cost
                    # cost = self.train_function(ins)[0]
                    cost, output, J = self.train_function(ins)
                    if np.isnan(cost):
                        LOG.debug(colors.cyan("loss runs into NaN, retry to train the neural network"))
                        self.retry = True
                        break

                if prev_cost == cost:
                    patient_list.append(cost)
                else:
                    start_flag = False
                    patient_list = []
                if len(patient_list) >= 50 and start_flag:
                    self.retry = True
                    LOG.debug("lost patient...")
                    break

                if np.isnan(cost):
                    self.retry = True
                    break

                # if cost < 10:
                #     import ipdb; ipdb.set_trace()
                #     predictions, mu, sigma = self.predict2(inputs)
                #     LOG.debug("Predicted mean: {}, sigma: {}".format(mean, std))
                #     LOG.debug("weight: {}".format(self.weight))
                self.cost_history.append([i, cost])
                LOG.debug("Epoch: {}, Loss: {}".format(i, cost))

            # if cost > 10:
            #     self.retry = True

        ins = [x, self.mean, self.std]
        retry_count = 0
        while retry_count < self._max_retry:
            init = tf.global_variables_initializer()
            utils.get_session().run(init)

            fit2_loop(ins)
            if self.retry is False:
                LOG.debug("Train neural network sucessfully, retry count: {}".format(retry_count))
                break

            retry_count += 1
            LOG.debug("Retry to train the neural network, retry count: {}".format(retry_count))

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:,0], cost_history[:, 1], loss_file_name)

    def predict2(self, inputs, steps_per_epoch=1):
        _inputs = ops.convert_to_tensor(inputs, tf.float32)
        if not self.built:
            self._need_compile = False
            self.build(_inputs)

        x = self.reshape(_inputs)
        output = self.model.predict(x, steps=steps_per_epoch)
        output = output.reshape(-1)
        diff = output[1:] - output[:-1]
        mean = diff.mean()
        std = diff.std()

        return output, float(mean), float(std)


    def save(self):
        pass

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
                 network_type=constants.NetworkType.PLAY
                 ):
        # fix random seed to 123
        seed = 123
        np.random.seed(seed)
        LOG.debug(colors.red("Make sure you are using the right random seed. currently seed is {}".format(seed)))

        self.plays = []
        self._nb_plays = nb_plays
        self._activation = activation
        self._input_dim = input_dim
        i = 1
        _weight = 1.0
        _width = 0.1
        width = 1
        for nb_play in range(nb_plays):
            # weight =  _weight / (i)
            # weight = 1.0
            if diff_weights is True:
                # weight =  2 * _weight / (i)
                # weight = i * _weight
                # xxxx: good weights
                # weight = 2 * _weight / i
                # weight = _weight * i / 10
                weight = 0.5 / (_width * i) # width range from (0.1, ... 0.1 * nb_plays)
            else:
                weight = 1.0

            LOG.debug("MyModel geneartes {} with Weight: {}".format(colors.red("Play #{}".format(i)), weight))

            play = Play(units=units,
                        batch_size=batch_size,
                        weight=weight,
                        width=width,
                        debug=debug,
                        activation=activation,
                        loss=None,
                        optimizer=None,
                        network_type=network_type,
                        name="play-{}".format(i),
                        timestep=timestep,
                        input_dim=input_dim)
            assert play._need_compile == False, colors.red("Play inside MyModel mustn't need compiled")
            self.plays.append(play)

            i += 1
        if optimizer is not None:
            self.optimzer = optimizers.get(optimizer)
        else:
            self.optimizer = None

    def fit(self, inputs, outputs, epochs=100, verbose=0, steps_per_epoch=1, loss_file_name="./tmp/mymodel_loss_history.csv", learning_rate=0.001, decay=0.):
        writer = utils.get_tf_summary_writer("./log/mse")

        inputs = ops.convert_to_tensor(inputs, tf.float32)
        outputs = ops.convert_to_tensor(outputs, tf.float32)

        for play in self.plays:
            if not play.built:
                play.build(inputs)

        x, y = self.plays[0].reshape(inputs, outputs)

        params_list = []
        model_inputs = []
        model_outputs = []
        feed_inputs = []
        feed_targets = []
        update_inputs = []

        for play in self.plays:
            inputs = play.model._layers[0].input
            outputs = play.model._layers[-1].output
            model_inputs.append(inputs)
            model_outputs.append(outputs)
            feed_inputs.append(inputs)

            for i in range(len(play.model.outputs)):
                shape = tf.keras.backend.int_shape(play.model.outputs[i])
                name = 'test{}'.format(i)
                target = tf.keras.backend.placeholder(
                    ndim=len(shape),
                    name=name + '_target',
                    dtype=tf.keras.backend.dtype(play.model.outputs[i]))

                feed_targets.append(target)

            update_inputs += play.model.get_updates_for(inputs)
            params_list += play.model.trainable_weights

        if self._nb_plays > 1:
            y_pred = tf.keras.layers.Average()(model_outputs)
        else:
            y_pred = model_outputs[0]

        loss = tf.keras.backend.mean(tf.math.square(y_pred - y))
        # decay: decay learning rate to half every 100 steps
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay)
        with tf.name_scope('training'):
            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=params_list,
                                                     loss=loss)
            updates += update_inputs

            training_inputs = feed_inputs + feed_targets
            train_function = tf.keras.backend.function(training_inputs,
                                                       [loss],
                                                       updates=updates)

        _x = [x for _ in range(self._nb_plays)]
        _y = [y for _ in range(self._nb_plays)]
        ins = _x + _y

        self.cost_history = []

        path = "/".join(loss_file_name.split("/")[:-1])
        writer.add_graph(tf.get_default_graph())
        loss_summary = tf.summary.scalar("loss", loss)

        for i in range(epochs):
            for j in range(steps_per_epoch):
                cost = train_function(ins)[0]
            self.cost_history.append([i, cost])
            # if i != 0  and i % 50 == 0:     # save weights every 50 epochs
            #     fname = "{}/epochs-{}/weights-mse.h5".format(path, i)
            #     self.save_weights(fname)
            LOG.debug("Epoch: {}, Loss: {}".format(i, cost))

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict(self, inputs, individual=False):

        import time
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
        i = 1
        for play in self.plays:
            LOG.debug("Play #{}, number of layer is: {}".format(i, play.number_of_layers))
            LOG.debug("Play #{}, weight: {}".format(i, play.weight))
            i += 1

    def fit2(self, inputs, mean, sigma, epochs=100, verbose=0,
             steps_per_epoch=1, loss_file_name="./tmp/mymodel_loss_history.csv",
             learning_rate=0.001, decay=0.):
        LOG.debug("Using MLE to fit")
        _inputs = inputs
        writer = utils.get_tf_summary_writer("./log/mle")
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        import time
        for play in self.plays:
            if not play.built:
                play.build(inputs)
        # from pool import _pool as pool
        # # import ipdb; ipdb.set_trace()
        # args_list = [(play, inputs) for play in self.plays]
        # self.plays = pool.starmap(utils.build_play, args_list)
        # pool.join()
        # pool.close()

        # import ipdb; ipdb.set_trace()
        #################### DO GRADIENT BY HAND HERE ####################
        def gradient_phi_cell(P):
            _P = tf.reshape(P, shape=(P.shape[0].value, -1))
            _diff = _P[:, 1:] - _P[:, :-1]

            x0 = tf.slice(_P, [0, 0], [1, 1])
            diff = tf.concat([x0, _diff], axis=1)

            p1 = tf.cast(tf.abs(diff) > 0., dtype=tf.float32)
            p2 = 1.0 - p1
            p3_list = []
            # TODO: multiple process here

            for j in range(1, _P.shape[1].value):
                p3_list.append(tf.reduce_sum(tf.cumprod(p2[:, j:], axis=1), axis=1))

            _p3 = tf.stack(p3_list, axis=1) + 1
            p3 = tf.concat([_p3, tf.constant(1.0, shape=(_p3.shape[0].value, 1), dtype=tf.float32)], axis=1)

            result = tf.multiply(p1, p3)
            return tf.reshape(result, shape=P.shape.as_list())

            # reshaped_P = tf.reshape(P, shape=(P.shape[0].value, -1))
            # diff = reshaped_P[:, 1:] - reshaped_P[:, :-1]
            # x0 = tf.slice(reshaped_P, [0, 0], [1, 1])
            # diff_ = tf.concat([x0, diff], axis=1)
            # result = tf.cast(tf.abs(diff_) >= 1e-7, dtype=tf.float32)
            # return tf.reshape(result, shape=P.shape)

        def gradient_nonlinear_layer(fZ, activation=None):
            LOG.debug("gradient nonlinear activation {}".format(activation))
            # ignore sample
            _fZ = tf.reshape(fZ, shape=fZ.shape.as_list()[1:])
            if activation is None:
                return tf.keras.backend.ones(shape=_fZ.shape)
            elif activation == 'tanh':
                ### might be a bug here
                ### we need to ensure the right epoch of fZ
                return (1.0 + _fZ) * (1.0 - _fZ)
            elif activation == 'relu':
                _fZ = tf.reshape(fZ, shape=fZ.shape.as_list()[1:])
                return tf.cast(_fZ >= 1e-8, dtype=tf.float32)
            else:
                raise Exception("activation: {} not support".format(activation))

        ################# FINISH GRADIENT BY HAND HERE ##################
        # x = self.plays[0].reshape(inputs)
        # NOTE(zxchen): don't move
        _x = [play.reshape(inputs) for play in self.plays]
        _x_feed_dict = {"input_{}:0".format(k+1) : _inputs.reshape(1, -1, self._input_dim) for k in range(self._nb_plays)}
        # import ipdb; ipdb.set_trace()
        self.mean = tf.Variable(mean, name="mean", dtype=tf.float32)
        self.sigma = tf.Variable(sigma, name="sigma", dtype=tf.float32)

        params_list = []
        model_inputs = []
        model_outputs = []
        feed_inputs = []
        feed_targets = []
        update_inputs = []

        target_mean = tf.keras.backend.placeholder(ndim=0, name="mean_target", dtype=tf.float32)
        target_sigma = tf.keras.backend.placeholder(ndim=0, name="mean_sigma", dtype=tf.float32)
        # feed_targets = [target_mean, target_mean]

        for play in self.plays:
            inputs = play.model._layers[0].input
            outputs = play.model._layers[-1].output
            model_inputs.append(inputs)
            model_outputs.append(outputs)
            feed_inputs.append(inputs)

            for i in range(len(play.model.outputs)):
                shape = tf.keras.backend.int_shape(play.model.outputs[i])
                name = 'test{}'.format(i)
                target = tf.keras.backend.placeholder(
                    ndim=len(shape),
                    name=name + '_target',
                    dtype=tf.keras.backend.dtype(play.model.outputs[i]))

                feed_targets.append(target)

            update_inputs += play.model.get_updates_for(inputs)
            params_list += play.model.trainable_weights

        if self._nb_plays > 1:
            y_pred = tf.keras.layers.Average()(model_outputs)
        else:
            y_pred = model_outputs[0]

        feed_inputs = model_inputs

        ##################### Prepare output of nonlinear #############################

        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  decay=decay)

        with tf.name_scope('training'):
            J_list = []
            _by_hand_list = []
            _by_tf_list = []
            last_nonlinear_outputs = []
            # import ipdb; ipdb.set_trace()
            ################### CALC J by hand #######################
            for idx in range(self._nb_plays):
                play = self.plays[idx]

                ###### Extract Layer's outputs ######
                reshaped_operator_layer = play.model.layers[1]
                operator_output = reshaped_operator_layer.output

                nonlinear_output = play.model.layers[2].output

                ######## Extract Layer's weights #######
                trainable_weights = play.model.trainable_weights

                # phi_weight = play.model.layers[0].cell.last_kernel
                # theta = play.model.layers[2].last_kernel
                # tilde_theta = play.model.layers[3].last_kernel

                phi_weight = play.model.layers[0].cell.kernel
                theta = play.model.layers[2].kernel
                tilde_theta = play.model.layers[3].kernel

                ######## HANDLE Layer's weights #########
                # import ipdb; ipdb.set_trace()
                start_tick = time.time()
                _gradient_tilde_theta = tf.transpose(tilde_theta, perm=[1, 0])
                end_tick = time.time()
                LOG.debug("Play #{} _gradient_tilde_theta cost: {} s".format(idx, end_tick-start_tick))

                start_tick = time.time()
                _theta = tf.tile(theta, multiples=[operator_output.shape[1].value, 1])
                end_tick = time.time()
                LOG.debug("Play #{} _theta cost: {} s".format(idx, end_tick-start_tick))

                start_tick = time.time()
                _gradient_theta = tf.multiply(_theta, gradient_nonlinear_layer(nonlinear_output, self._activation))
                end_tick = time.time()
                LOG.debug("Play #{} _gradient_theta cost: {} s".format(idx, end_tick-start_tick))

                start_tick = time.time()
                gradient_nonlinear = tf.matmul(_gradient_theta, tilde_theta)
                end_tick = time.time()
                LOG.debug("Play #{} gradient_nonlinear cost: {} s".format(idx, end_tick-start_tick))

                start_tick = time.time()
                _gradient_phi = gradient_phi_cell(play.layers[1].output)
                end_tick = time.time()
                LOG.debug("Play #{} _gradient_phi cost: {} s".format(idx, end_tick-start_tick))

                start_tick = time.time()
                gradient_phi = tf.multiply(_gradient_phi, phi_weight)
                end_tick = time.time()
                LOG.debug("Play #{} gradient_phi cost: {} s".format(idx, end_tick-start_tick))

                start_tick = time.time()
                gradient_J = tf.multiply(gradient_phi, gradient_nonlinear)
                end_tick = time.time()
                LOG.debug("Play #{} gradient_J cost: {} s".format(idx, end_tick-start_tick))
                #############################################
                # import ipdb; ipdb.set_trace()
                auto_gradient_nonlinear = tf.keras.backend.gradients(play.model.layers[3].output, play.model.layers[2].input)
                auto_gradient_phi = tf.keras.backend.gradients(play.model.layers[0].output, play.model.layers[0].input)
                auto_gradient_J = tf.keras.backend.gradients(play.model.layers[3].output, play.model.layers[0].input)

                # import ipdb; ipdb.set_trace()
                ###############################################
                _by_hand_list.append([gradient_nonlinear, gradient_phi, gradient_J])
                _by_tf_list.append([auto_gradient_nonlinear[0], auto_gradient_phi[0], auto_gradient_J[0]])
                J_list.append(gradient_J)

            diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]

            ###################### Calculate J by Tensorflow ###############################
            for i in range(self._nb_plays):
                model_outputs[i] = tf.reshape(model_outputs[i], shape=model_inputs[i].shape)

            y_pred = tf.reshape(y_pred, shape=model_inputs[0].shape)

            J = tf.keras.backend.gradients(model_outputs, model_inputs)
            for i in range(self._nb_plays):
                J[i] = tf.reshape(J[i], shape=(1, -1, 1))

            J = tf.reduce_mean(tf.concat(J, axis=-1), axis=-1,keepdims=True) / self._nb_plays
            ###################### Calculate J by hand ###############################
            # (T * 1)
            J_by_hand = tf.reduce_mean(tf.concat(J_list, axis=-1), axis=-1, keepdims=True) / self._nb_plays
            normalized_J = tf.clip_by_value(tf.abs(J_by_hand), clip_value_min=1e-7, clip_value_max=1e9)
            ################################################################################

            # TODO: support derivation for p0

            # _loss = tf.keras.backend.square((diff-self.mean)/self.sigma)/2.0 - tf.keras.backend.log(detJ[:, 1:, :])
            # BUGS: something wrong here
            # _loss = tf.keras.backend.square((diff-self.mean)/self.sigma)/2.0 - tf.keras.backend.log(normalized_J[:, 1:, :])
            _loss = tf.keras.backend.square(diff/sigma)/2.0 - tf.keras.backend.log(normalized_J[:, 1:, :])
            loss = tf.keras.backend.mean(_loss)

            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=params_list,
                                                     loss=loss)

            updated_weights = updates[2::3]
            # import ipdb; ipdb.set_trace()

            training_inputs = feed_inputs + feed_targets
            train_function = tf.keras.backend.function(training_inputs,
                                                       # [loss, _by_hand_list, _by_tf_list, updated_weights, J, J_by_hand],
                                                       [loss, J, J_by_hand],
                                                       # [loss],
                                                       updates=updates)

        # _y = [y for _ in range(self._nb_plays)]
        # import ipdb; ipdb.set_trace()
        _y = [self.mean, self.sigma]
        ins = _x + _y
        utils.init_tf_variables()

        writer.add_graph(tf.get_default_graph())
        self.cost_history = []
        steps_per_epoch = 1
        once = True
        cost = -1

        # updated_weights_res = sess.run(updated_weights, feed_dict=_x_feed_dict)

        # k = 0
        # batch_assign_tuples = []
        # for play in self.plays:
        #     batch_assign_tuples.append(
        #         (
        #             play.layers[0].last_kernel,
        #             # tf.keras.backend.get_value(play.layers[0].kernel)
        #             updated_weights_res[k*5+0]
        #         )
        #     )
        #     batch_assign_tuples.append(
        #         (
        #             play.layers[2].last_kernel,
        #             updated_weights_res[k*5+1]
        #             # tf.keras.backend.get_value(play.layers[2].kernel)
        #         )
        #     )
        #     batch_assign_tuples.append(
        #         (
        #             play.layers[3].last_kernel,
        #             updated_weights_res[k*5+3]
        #             # tf.keras.backend.get_value(play.layers[3].kernel)
        #         )
        #     )


        for i in range(epochs):
            for j in range(steps_per_epoch):
                # import ipdb; ipdb.set_trace()

                # k = 0
                # batch_assign_tuples = []
                # print(colors.yellow("############################################################################################"))
                # for play in self.plays:
                #     old_phi_weight = tf.keras.backend.get_value(play.model.layers[0].last_kernel)
                #     old_theta = tf.keras.backend.get_value(play.model.layers[2].last_kernel)
                #     old_tilde_theta = tf.keras.backend.get_value(play.model.layers[3].last_kernel)
                #     curr_phi_weight = tf.keras.backend.get_value(play.model.layers[0].kernel)
                #     curr_theta = tf.keras.backend.get_value(play.model.layers[2].kernel)
                #     curr_tilde_theta = tf.keras.backend.get_value(play.model.layers[3].kernel)

                #     old_theta = old_theta.reshape(-1)
                #     old_tilde_theta = old_tilde_theta.reshape(-1)
                #     curr_theta = curr_theta.reshape(-1)
                #     curr_tilde_theta = curr_tilde_theta.reshape(-1)
                #     print("{} Before assign phiweight: {}".format(colors.red("Play #{}".format(k)), old_phi_weight))
                #     print("{} After assign phi_weight: {}".format(colors.red("Play #{}".format(k)), curr_phi_weight))
                #     print("{} Before assign theta: {}".format(colors.red("Play #{}".format(k)), old_theta))
                #     print("{} After assign theta: {}".format(colors.red("Play #{}".format(k)), curr_theta))
                #     print("{} Before assign tilde theta: {}".format(colors.red("Play #{}".format(k)), old_tilde_theta))
                #     print("{} After  assign tilde theta: {}".format(colors.red("Play #{}".format(k)), curr_tilde_theta))
                #     print("{} Before  results: {}".format(colors.red("Play #{}".format(k)), (old_tilde_theta * old_theta).sum()))
                #     print("{} After  results: {}".format(colors.red("Play #{}".format(k)), (curr_tilde_theta * curr_theta).sum()))
                #     print("------------------------------------------------------------------------------------------")

                #     batch_assign_tuples.append(
                #         (
                #             play.model.layers[0].cell.last_kernel,
                #             tf.keras.backend.get_value(play.model.layers[0].cell.kernel)
                #         )
                #     )
                #     batch_assign_tuples.append(
                #         (
                #             play.model.layers[2].last_kernel,
                #             tf.keras.backend.get_value(play.model.layers[2].kernel)
                #         )
                #     )
                #     batch_assign_tuples.append(
                #         (
                #             play.model.layers[3].last_kernel,
                #             tf.keras.backend.get_value(play.model.layers[3].kernel)
                #         )
                #     )
                #     k += 1

                # tf.keras.backend.batch_set_value(batch_assign_tuples)
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                # cost, by_hand_list, by_tf_list, updated_weights_res, J_res, J_by_hand_res = train_function(ins)
                cost, J_res, J_by_hand_res = train_function(ins)
                # cost = train_function(ins)[0]

                # m = 0
                # # import ipdb; ipdb.set_trace()
                # for _a, _b in zip(by_hand_list, by_tf_list):
                #     print("--------------------------------------------------------------------------------")
                #     print("{}".format(colors.red("index is: {}".format(m))))

                #     delta = np.abs(_b[0].reshape(-1) - _a[0].reshape(-1))
                #     argmax = np.argmax(delta)
                #     print("_a[0][{}]: {}".format(argmax, _a[0].reshape(-1)[argmax]))
                #     print("_b[0][{}]: {}".format(argmax, _b[0].reshape(-1)[argmax]))
                #     print("delta, mean: {}, max: {}".format(delta.mean(), delta.max()))

                #     if delta.max() >= 0.1:
                #         print(colors.red("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX theta max XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"))
                #         once = False
                #     if not np.allclose(_a[0], _b[0], rtol=1e-5, atol=1e-8):
                #         print(colors.red("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX theta  allclose XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"))
                #         once = False

                #     # print("by_hand[1]: ", _a[1].reshape(-1).tolist())
                #     # print("by_tf[1]: ", _b[1].reshape(-1).tolist())
                #     delta1 = np.abs(_b[1].reshape(-1)-_a[1].reshape(-1))
                #     print(colors.red("delta1, max {}".format(delta1.max())))
                #     if not np.allclose(_a[1].reshape(-1), _b[1].reshape(-1), rtol=1e-5, atol=1e-8):
                #         print(colors.red("XXXXXXXXXXXXXXXXXXXXXXXXXXXX      Phi      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"))
                #         once = False

                #     # print("by_hand[2]: ", _a[2].reshape(-1).tolist())
                #     # print("by_tf[2]: ", _b[2].reshape(-1).tolist())

                #     # delta2 = np.abs(_b[2].reshape(_a[2].shape)-_a[2]).max()
                #     # print(colors.red("delta2, max {}".format(delta2)))
                #     # print(colors.red("XXXXXXXXXXXXXXXXX   DELTA2: {}".format(delta2.reshape(-1).tolist())))
                #     # if not np.allclose(_a[2], _b[2].reshape(_a[2].shape), rtol=1e-2, atol=1e-7):
                #     #     print(colors.red("XXXXXXXXXXXXXXXXXX   Gradient   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"))
                #     #     once = False

                #     m += 1

                # # print(colors.yellow("J_res: {}".format(J_res)))
                # # print(colors.yellow("J_by_handres: {}".format(J_by_hand_res)))
                # # import ipdb; ipdb.set_trace()
                delta_J = np.abs(J_res.reshape(-1) - J_by_hand_res.reshape(-1))
                print(colors.yellow("diff mean: {}, max: {}".format(delta_J.mean(), np.max(delta_J))))
                if not np.allclose(J_by_hand_res.reshape(-1), J_res.reshape(-1), rtol=1e-2, equal_nan=True, atol=1e-3):
                    print(colors.red("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"))
                    once = False

                print(colors.yellow("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"))
                # if not once:
                #     import ipdb; ipdb.set_trace()

                # for J1, J2 in zip(J_res, J_list_res):
                #     if not np.allclose(J1, J2, rtol=1e-5, atol=1e-3):
                #     # if True:
                #         print("J1: {}".format(J1))
                #         print("J2: {}".format(J2))
                #         print("Diff: {}".format(np.abs(J1-J2)))
                #         import ipdb; ipdb.set_trace()

                        # raise Exception("Not the same....")
                # print("detJ: {}".format(detJ_res))
                # print("J_list: {}".format(J_list_res))
                # print("intral_res_list: {}".format(intral_res_liss))
                # print("aa_list: {}".format(aa_list_res))
            LOG.debug("Epoch: {}, Loss: {}".format(i, cost))
            self.cost_history.append([i, cost])


        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict2(self,  inputs):
        _inputs = ops.convert_to_tensor(inputs, tf.float32)
        for play in self.plays:
            self._need_compile = False
            if not play.built:
                play.build(_inputs)

        x = self.plays[0].reshape(inputs)
        outputs = []
        for play in self.plays:
            outputs.append(play.predict(x))
        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        diff = prediction[1:] - prediction[:-1]
        mean = diff.mean()
        std = diff.std()
        LOG.debug("mean: {}, std: {}".format(mean, std))
        return prediction, float(mean), float(std)

    def save_weights(self, fname):
        suffix = fname.split(".")[-1]
        for play in self.plays:
            path = "{}/{}plays/{}.{}".format(fname[:-3], len(self.plays), play._name, suffix)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()
            LOG.debug(colors.cyan("Saving {}'s Weights to {}".format(play._name, path)))
            play.model.save_weights(path)

        LOG.debug(colors.cyan("riting input shape into disk..."))
        with open("{}/{}plays/input_shape.txt".format(fname[:-3], len(self.plays)), "w") as f:
            f.write(":".join(map(str, self.plays[0]._batch_input_shape.as_list())))

    def load_weights(self, fname):
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

        for play in self.plays:
            if not play.built:
                play._batch_input_shape = tf.TensorShape(shape)
                play._preload_weights = True
                play.build()
            path = "{}/{}.{}".format(dirname, play._name, suffix)
            play.model.load_weights(path, by_name=False)
            LOG.debug(colors.red("Set Weights for {}".format(play._name)))


def gradient_phi_cell(P):
    _P = tf.reshape(P, shape=(P.shape[0].value, -1))
    _diff = _P[:, 1:] - _P[:, :-1]

    x0 = tf.slice(_P, [0, 0], [1, 1])
    diff = tf.concat([x0, _diff], axis=1)

    p1 = tf.cast(tf.abs(diff) >= 1e-7, dtype=tf.float32)
    p2 = 1.0 - p1
    p3_list = []
    for j in range(1, _P.shape[1].value):
        p3_list.append(tf.reduce_sum(tf.cumprod(p2[:, j:], axis=1), axis=1))
    _p3 = tf.stack(p3_list, axis=1) + 1
    p3 = tf.concat([_p3, tf.constant(1.0, shape=(_p3.shape[0].value, 1), dtype=tf.float32)], axis=1)

    result = tf.multiply(p1, p3)
    return tf.reshape(result, shape=P.shape.as_list()[1:])



if __name__ == "__main__":
    # set random seed to make results reproducible
    tf.random.set_random_seed(123)
    np.random.seed(123)

    ## Test
    # a = tf.keras.backend.ones(shape=[1,2,3])
    # init = tf.global_variables_initializer()
    # sess.run(init)
    aa = np.array([1])
    aa = aa.reshape([1, 1])
    a = tf.constant(aa, dtype=tf.float32, name="a")
    phi = Phi(a, 0)
    g_phi = tf.gradients(phi, [a], name='phi')
    print(sess.run(g_phi))
    a = tf.constant(aa, dtype=tf.float32, name="a")
    phi = Phi(a)
    g_phi = tf.gradients(phi, [a], name='phi')
    print(sess.run(g_phi))

    # import ipdb; ipdb.set_trace()

    # print(sess.run(a))
    writer = utils.get_tf_summary_writer("./log/")
    aa = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    aa = aa.reshape([-1, 1])
    bb = np.array([2, 3])

    bb = bb.reshape([1, -1])
    cc = np.array([4, 5])
    cc = cc.reshape([-1, 1])

    a = tf.constant(aa, dtype=tf.float32, name="a")
    b = tf.constant(bb, dtype=tf.float32, name="b")
    c = tf.constant(cc, dtype=tf.float32, name="c")
    # d = a * b
    d = tf.multiply(a, b, name="d")
    # e = tf.keras.backend.dot(d, c)
    e = tf.matmul(d, c, name="e")
    # c_reshape = tf.reshape(c, shape=(-1, ))
    # c_t = tf.transpose(c, perm=[1, 0])
    # cn = tf.tile(c_t, [2, 1])

    # c = tf.keras.backend.dot(a,  b)
    g1 = tf.gradients(d, [a], name="g1")
    g2 = tf.gradients(e, [d], name="g2")

    g3 = tf.gradients(e, [a], name="g3")

    writer.add_graph(tf.get_default_graph())


    # print(sess.run(g1))
    # print(sess.run(g2))
    # print(sess.run(g3))
    # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    # a = tf.constant([[1], [2], [3], [4], [5], [6]], dtype=tf.float32)
    # b= tf.constant([[1], [2], [3], [4], [5], [6]], dtype=tf.float32)
    # a = tf.constant([[1, 2, 3]], dtype=tf.float32)
    # b = tf.constant([[1], [2], [3]], dtype=tf.float32)
    # print(sess.run(a * b))
    # import ipdb; ipdb.set_trace()

    # a = tf.constant([[[1], [2], [4]]], dtype=tf.float32)
    # b = tf.constant([[2, 3]], dtype=tf.float32)
    # print(sess.run(a * b))
    # print(sess.run(b * a))
    # import ipdb; ipdb.set_trace()
    # a = tf.constant([0, 2, 3, -1, 3, 4, 0], dtype=tf.float32)
    # zero = tf.constant(0, dtype=tf.float32)
    # where = tf.not_equal(a, zero)


    # import ipdb; ipdb.set_trace()
    # a_np = np.array([0, 2, 3, -1, 3, 4, 0])
    # a_np = a_np.reshape(1, -1, 1)
    # a = tf.constant(a_np, dtype=tf.float32)
    # b = tf.constant(2., shape=(1, 1), dtype=tf.float32)
    # print(sess.run(a * a))
    # a_slice = tf.slice(a, [0, 0], [1, 1])
    # a_stack = tf.concat([a_slice, a], axis=1)
    # print("HELLO WORLD")
    # import ipdb; ipdb.set_trace()
    # from tensorflow.python.ops import standard_ops
    # sess = utils.get_session()

    # a = tf.constant([[[1, 2],
    #                   [3, 4],
    #                   [5, 6]]])
    # a = tf.reshape(a, shape=(1, -1, 1))
    # b = tf.constant([[1, 2], [3,4]])
    # b = tf.reshape(b, shape=(1, 4))
    # c = tf.constant([[1, 2, 3, 4]])
    # # # c = standard_ops.tensordot(a, b, [[2], [0]])
    # d = a * b
    # e = d + c
    # f = tf.constant([[1, 2, 3, 4]])
    # g = e * f
    # h = tf.reshape(f, shape=(4, 1))

    # i = standard_ops.tensordot(e, h, [[2], [0]])
    # k = tf.constant(1, shape=(1, 1))
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # print("a.shape: ", a.shape)
    # print("b.shape: ", b.shape)
    # print("c.shape: ", c.shape)
    # print("d.shape: ", d.shape)
    # print("e.shape: ", e.shape)
    # print("f.shape: ", f.shape)
    # print("g.shape: ", g.shape)
    # print("h.shape: ", h.shape)
    # print("i.shape: ", i.shape)
    # print("j.shape: ", tf.reduce_sum(g, axis=2))
    # print("a: ", sess.run(a))
    # print("b: ", sess.run(b))
    # print("c: ", sess.run(c))
    # print("d: ", sess.run(d))
    # print("e: ", sess.run(e))
    # print("f: ", sess.run(f))
    # print("g: ", sess.run(g))
    # print("h: ", sess.run(h))
    # print("i: ", sess.run(i))
    # print("j: ", sess.run(tf.reduce_sum(g, axis=2)))
    # print("k: ", sess.run(tf.reduce_sum(g, axis=2)+k))
    # import ipdb; ipdb.set_trace()

    # phi_cell = PhiCell(weight=1.0, width=1.0, debug=True)
    # _x = np.array([-2.5, -1.5, -0.5, -0.7, 0.5, 1.5])
    # outputs = phi_cell(_x, [0])

    # init = tf.global_variables_initializer()
    # sess.run(init)

    # LOG.debug("outputs: {}".format(sess.run(outputs)))
    # inputs = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 0.33, 0.1, 0, 0.21, -1.5, 0.7, 0.9, 1.5, -0.4, 1, -0.15, 2])


    _x = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 0.33, 0.1, 0, 0.21, -1.5, 0.7, 0.9, 1.5, -0.4, 1, -0.15, 2])
    # _x = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 0.33, 0.1])
    # _x = _x.reshape((1, -1, 1))
    _x = _x.reshape((1, -1, 2))
    x = ops.convert_to_tensor(_x, dtype=tf.float32)
    LOG.debug("x.shape: {}".format(x.shape))
    initial_state = None
    layer = Operator(debug=True, weight=1)
    y = layer(x, initial_state)
    g = tf.gradients(y, [x])
    init = tf.global_variables_initializer()
    sess.run(init)

    y_res = sess.run(y)
    LOG.debug("y: {}".format(y_res))

    g_by_hand = gradient_phi_cell(y)


    LOG.debug("g: {}".format(sess.run(g)))
    LOG.debug("g_by_hand: {}".format(sess.run(g_by_hand)))

    # import ipdb; ipdb.set_trace()

    # _y_true = y_res

    # # _x = _x.reshape((1, -1, 10))
    # # x = ops.convert_to_tensor(_x, dtype=tf.float32)
    # # layer = Operator(debug=True, weight=3)
    # # y = layer(x, initial_state)
    # # init = tf.global_variables_initializer()
    # # sess.run(init)
    # # y_res = sess.run(y)

    # # LOG.debug("all close: {}.".format(np.allclose(y_res.reshape(-1), _y_true.reshape(-1))))

    # batch_size = 2
    # epochs = 2500 // 10
    # steps_per_epoch = 20
    # units = 2
    # _x = _x.reshape(-1)
    # _y_true = _y_true.reshape(-1)

    # import time
    # start = time.time()
    # play = Play(batch_size=batch_size,
    #             units=units,
    #             activation=None)
    # play.fit(_x, _y_true, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # end = time.time()
    # LOG.debug("time cost: {}s".format(end-start))
    # import ipdb; ipdb.set_trace()
    # score = play.evaluate(_x, _y_true)
    # LOG.debug("score: {}".format(score))
    # LOG.debug("weight: {}".format(play.weight))

    # import trading_data as tdata

    # batch_size = 10
    # # epochs = 100 // batch_size
    # epochs = 10000 // batch_size
    # steps_per_epoch = batch_size
    # units = 4

    # LOG.debug(colors.red("Test Operator"))
    # fname = constants.FNAME_FORMAT["operators"].format(method="sin", weight=1, width=1)

    # inputs, outputs = tdata.DatasetLoader.load_data(fname)
    # LOG.debug("timestap is: {}".format(inputs.shape[0]))

    # batch_size = 20
    # epochs = 5000 // batch_size
    # steps_per_epoch = batch_size
    # units = 10

    # play = Play(batch_size=batch_size,
    #             units=units,
    #             activation=None,
    #             network_type=constants.NetworkType.OPERATOR)

    # play.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)

    # LOG.debug("number of layer is: {}".format(play.number_of_layers))
    # LOG.debug("weight: {}".format(play.weight))

    LOG.debug(colors.red("Test Play"))
    # batch_size = 1
    # timestep = 50
    # input_dim = 10
    # units = 20
    # epochs = 1500 // batch_size
    # steps_per_epoch = batch_size

    # fname = constants.FNAME_FORMAT["plays"].format(method="sin", weight=1, width=1, points=500)

    # inputs, outputs = tdata.DatasetLoader.load_data(fname)
    # length = 500
    # inputs, outputs = inputs[:length], outputs[:length]

    # LOG.debug("timestap is: {}".format(inputs.shape[0]))

    # import time
    # start = time.time()
    # play = Play(batch_size=batch_size,
    #             units=units,
    #             activation="tanh",
    #             network_type=constants.NetworkType.PLAY,
    #             loss='mse',
    #             debug=True,
    #             timestep=timestep,
    #             input_dim=input_dim)

    # play.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # end = time.time()
    # LOG.debug("time cost: {}s".format(end-start))

    # LOG.debug("number of layer is: {}".format(play.number_of_layers))
    # LOG.debug("weight: {}".format(play.weight))

    # a = tf.constant([1, 2, 3, 4], dtype=tf.float32)
    # b = tf.reshape(a, shape=(1, -1, 1))
    # c = tf.cumprod(b, axis=1)
    # d = b * c
    # print("c: ", utils.get_session().run(c))
    # print("d: ", utils.get_session().run(d))


    LOG.debug(colors.red("Test multiple plays"))

    fname = "./new-dataset/models/method-sin/activation-None/state-0/mu-0/sigma-0/units-1/nb_plays-1/points-1000/input_dim-1/base.csv"
    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    length = 500
    # inputs = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 0.33, 0.1, 0, 0.21, -1.5, 0.7, 0.9, 1.5, -0.4, 1, -0.15, 2])

    inputs, outputs = inputs[:length], outputs[:length]
    print(inputs)
    # layer = Operator(debug=True, weight=1.0)
    # inputs = inputs.reshape([1, -1, 1])
    # inputs_ = ops.convert_to_tensor(inputs, dtype=tf.float32)
    # outputs = layer(inputs_)
    # init = tf.global_variables_initializer()
    # sess.run(init)

    # print(sess.run(outputs))
    # print("========================================")

    input_dim = 10
    timestep = length // input_dim
    LOG.debug("timestap is: {}".format(inputs.shape[0]))
    inputs = inputs.reshape(-1)
    units = 1
    epochs = 1000
    steps_per_epoch = 1
    mu = 0
    # sigma = 0.01
    sigma = 2
    nb_plays = 20
    __nb_plays__ = 5
    __units__ = 1
    # __activation__ = 'tanh'
    __activation__ = None
    # __activation__ = 'relu'
    import time
    start = time.time()
    agent = MyModel(input_dim=input_dim,
                    timestep=timestep,
                    units=__units__,
                    activation=__activation__,
                    nb_plays=__nb_plays__,
                    debug=True)

    agent.fit2(inputs, mu, sigma, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    # prediction, mu_prediction, sigma_prediction = agent.predict2(inputs)
    # prediction = agent.predict(inputs)
    # point_ = prediction.shape[-1]
    # fname = constants.FNAME_FORMAT["G_predictions"].format(method="sin", weight=1, width=1, points=point_, mu=mu_prediction, sigma=sigma_prediction, activation='tanh',
    #                                                        units=units, loss='mse')
    # tdata.DatasetSaver.save_data(inputs, prediction, fname)

    # LOG.debug("print weights info")
    # agent.weights


    # LOG.debug(colors.red("Test play with MLE"))
    # batch_size = 10
    # epochs = 10000 // batch_size
    # steps_per_epoch = batch_size
    # units = 1
    # points = 100

    # mu = 1
    # sigma = 0.01
    # method = "sin"
    # width = 1
    # weight = 1
    # inputs = tdata.DatasetGenerator.systhesis_markov_chain_generator(points=points, mu=mu, sigma=sigma)
    # # fname = constants.FNAME_FORMAT["plays_noise"].format(method="sin", weight=1, width=1, mu=mu, sigma=sigma)
    # # # fname = constants.FNAME_FORMAT["operators_noise"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma)
    # # inputs, outputs = tdata.DatasetLoader.load_data(fname)

    # length = 100
    # inputs, outputs = inputs[:length], outputs[:length]

    # play = Play(batch_size=batch_size,
    #             units=units,
    #             # activation='tanh',
    #             activation=None,
    #             network_type=constants.NetworkType.PLAY,
    #             loss=None,
    #             debug=False)

    # import time
    # start = time.time()

    # play.fit2(inputs, mu, sigma, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch)
    # end = time.time()
    # LOG.debug("time cost: {}s".format(end-start))
    # # import ipdb; ipdb.set_trace()
    # predictions, mean, std = play.predict2(inputs)
    # LOG.debug("Predicted mean: {}, sigma: {}".format(mean, std))
    # LOG.debug("weight: {}".format(play.weight))
    # # import ipdb; ipdb.set_trace()
    # print("End")
