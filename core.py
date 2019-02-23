import os
import tensorflow as tf
# tf.enable_eager_execution()

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

sess = utils.get_session()


def Phi(x, width=1.0):
    """
    Phi(x) = x         , if x > 0
           = x + width , if x < - width
           = 0         , otherwise
    """
    return tf.maximum(x, 0) + tf.minimum(x+width, 0)


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

        # TODO: try non negative constraint
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.hysteretic_func = hysteretic_func
        self.input_dim = input_dim

    def build(self, input_shape):
        if self.debug:
            LOG.debug("Initialize *weight* as pre-defined: {} ....".format(self._weight))
            self.kernel = tf.Variable([[self._weight]], name="weight", dtype=tf.float32)
            # if constants.DEBUG_INIT_TF_VALUE:
            #     self.kernel = self.kernel.initialized_value()

            self._trainable_weights.append(self.kernel)
        else:
            LOG.debug("Initialize *weight* randomly...")
            assert self.units == 1, "Phi Cell unit must be equal to 1"
            self.kernel = self.add_weight(
                "weight",
                shape=(self.units, self.units),
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

        outputs_ = tf.multiply(self._inputs, self.kernel)

        # NOTE: unroll method, can we use RNN method ?
        outputs = [self._state]
        for i in range(outputs_.shape[-1].value):
            output = tf.add(Phi(tf.subtract(outputs_[0][i], outputs[-1])), outputs[-1])
            outputs.append(output)
        outputs = ops.convert_to_tensor(outputs[1:], dtype=tf.float32)
        state = outputs[-1]
        outputs = tf.reshape(outputs, shape=self._inputs.shape)
        state = tf.reshape(state, shape=(1, 1))

        return outputs, [state]


class Operator(RNN):
    def __init__(self, weight=1.0, width=1.0, debug=False):
        cell = PhiCell(
            weight=weight,
            width=width,
            debug=debug
            )
        super(Operator, self).__init__(
            cell=cell,
            return_sequences=True,
            # return_state=True
            return_state=False,
            stateful=True,
            unroll=False
            )

    def call(self, inputs, initial_state=None):
        return super(Operator, self).call(inputs, initial_state=initial_state)


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
        if self._debug:
            LOG.debug("init mydense kernel/bias as pre-defined")
            _init_kernel = np.array([[1 for i in range(self.units)]])
            self.kernel = tf.Variable(_init_kernel, name="weight", dtype=tf.float32)
            self._trainable_weights.append(self.kernel)
            if self.use_bias is True:
                self.bias = tf.Variable(0, name="bias", dtype=tf.float32)
                self._trainable_weights.append(self.bias)

        else:
            self.kernel = self.add_weight(
                "kernel",
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

        outputs = inputs * self.kernel
        if self.use_bias:
            outputs += self.bias

        if self.activation is not None:
            outputs =  self.activation(outputs)

        return outputs


class MySimpleDense(Dense):
    def __init__(self, **kwargs):
        self._debug = kwargs.pop("debug", False)

        super(MySimpleDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.units == 1
        if self._debug is True:
            LOG.debug("init mysimpledense kernel/bias as pre-defined")
            _init_kernel = np.array([1 for _ in range(input_shape[-1].value)])
            _init_kernel = _init_kernel.reshape(-1, 1)
            self.kernel = tf.Variable(_init_kernel, name="kernel", dtype=tf.float32)
            self._trainable_weights.append(self.kernel)

            if self.use_bias:
                self.bias = tf.Variable((0,), name="bias", dtype=tf.float32)
                self._trainable_weights.append(self.bias)
        else:
            super(MySimpleDense, self).build(input_shape)

        self.built = True

    def call(self, inputs):
        return super(MySimpleDense, self).call(inputs)


class Play(object):
    def __init__(self, inputs=None,
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
                 name="play"):

        if debug:
            self._weight = weight
            self._width = width
            self._debug = debug

        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer

        self.batch_size = batch_size
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
            _inputs = ops.convert_to_tensor(inputs, tf.float32)

            if _inputs.shape.ndims == 1:
                length = _inputs.shape[-1].value
                timesteps = length // self.batch_size
                if timesteps * self.batch_size != length:
                    raise Exception("The batch size cannot be divided by the length of input sequence.")
                self._batch_input_shape = tf.TensorShape([1, timesteps, self.batch_size])
            else:
                raise Exception("dimension of inputs must be equal to 1")


        length = self._batch_input_shape[1].value * self._batch_input_shape[2].value
        timesteps = self._batch_input_shape[1].value

        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.InputLayer(batch_size=1,
                                                  input_shape=self._batch_input_shape[1:]))
        self.model.add(Operator(weight=getattr(self, "_weight", None),
                                width=getattr(self, "_width", None),
                                debug=getattr(self, "_debug", False)))
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
                y = tf.reshape(outputs, shape=(1, -1, 1))

            elif self._network_type == constants.NetworkType.PLAY:
                y = tf.reshape(outputs, shape=(1, -1, 1))

                # y = tf.manip.tile(outputs, [self.units])
                # y = tf.reshape(y, shape=(1, -1, self.units))
                # y = tf.reshape(outputs, shape=(1, -1))
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


class MyModel(object):
    def __init__(self, nb_plays=1,
                 units=1,
                 batch_size=1,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 activation='tanh',
                 # loss='mse',
                 optimizer='adam'):

        self.plays = []
        self._nb_plays = nb_plays
        self._pool = pool.ProcessPool()
        i = 1
        for nb_play in range(nb_plays):
            weight = np.random.uniform(0., 3.)
            LOG.debug("MyModel geneartes Play {} with Weight: {}".format(i, weight))

            play = Play(units=units,
                        batch_size=batch_size,
                        weight=weight,
                        width=width,
                        debug=debug,
                        activation=activation,
                        loss=None,
                        optimizer=None,
                        network_type=constants.NetworkType.PLAY,
                        name="play-{}".format(i))
            self.plays.append(play)
            i += 1

        self.optimzer = optimizers.get(optimizer)

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

        while i < epochs:
            i += 1
            for j in range(steps_per_epoch):
                cost = train_function(ins)[0]
            self.cost_history.append([i, cost])
            if i % 50 == 0:     # save weights every 50 epochs
                fname = "{}/epochs-{}/weights-mse.h5".format(path, i)
                self.save_weights(fname)
            LOG.debug("Epoch: {}, Loss: {}".format(i, cost))

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict(self, inputs):
        import time
        inputs = ops.convert_to_tensor(inputs, tf.float32)

        for play in self.plays:
            if not play.built:
                play.build(inputs)
        # args_list = [(play, inputs) for play in self.plays]
        # self._pool.starmap(self._build, args_list)
        # self._pool.join()

        x = self.plays[0].reshape(inputs)
        outputs = []

        for play in self.plays:
            start = time.time()
            outputs.append(play.predict(x))
            end = time.time()
            LOG.debug("play {} cost time {} s".format(play._name, end-start))

        # args_list = [(play, x) for play in self.plays]
        # self._pool.startmap(self._predict, args_list)
        # self._pool.join()

        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        return prediction

    @property
    def weights(self):
        i = 1
        for play in self.plays:
            LOG.debug("Play #{}, number of layer is: {}".format(i, play.number_of_layers))
            LOG.debug("Play #{}, weight: {}".format(i, play.weight))
            i += 1

    def fit2(self, inputs, mean, sigma, epochs=100, verbose=0,
             steps_per_epoch=1, loss_file_name="./tmp/mymodel_loss_history.csv"):
        writer = utils.get_tf_summary_writer("./log/mle")
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        import time
        for play in self.plays:
            if not play.built:
                play._need_compile = False
                start = time.time()
                play.build(inputs)
                end = time.time()
                LOG.debug("play {} time cost: {} s".format(play._name, end-start))
        # x, y = self.plays[0].reshape(inputs, outputs)
        x = self.plays[0].reshape(inputs)

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

        self.optimizer = tf.keras.optimizers.Adam()

        with tf.name_scope('training'):

            J = tf.keras.backend.gradients(y_pred, model_inputs)
            detJ = tf.reshape(tf.keras.backend.abs(J[0]), shape=y_pred.shape)
            detJ = tf.keras.backend.clip(detJ, min_value=1e-5, max_value=1e9)

            diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]

            _loss = tf.keras.backend.square((diff-self.mean)/self.sigma)/2.0 - tf.keras.backend.log(detJ[:, 1:, :])
            loss = tf.keras.backend.mean(_loss)

            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=params_list,
                                                     loss=loss)
            updates += update_inputs

            training_inputs = feed_inputs + feed_targets
            train_function = tf.keras.backend.function(training_inputs,
                                                       [loss],
                                                       updates=updates)

        _x = [x for _ in range(self._nb_plays)]
        # _y = [y for _ in range(self._nb_plays)]

        _y = [self.mean, self.sigma]
        ins = _x + _y
        utils.init_tf_variables()

        writer.add_graph(tf.get_default_graph())
        self.cost_history = []
        while i < epochs:
            i += 1
            for j in range(steps_per_epoch):
                cost = train_function(ins)[0]
            self.cost_history.append([i, cost])
            LOG.debug("Epoch: {}, Loss: {}".format(i, cost))

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

if __name__ == "__main__":
    # set random seed to make results reproducible

    # tf.random.set_random_seed(123)
    # np.random.seed(123)

    ## Test
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

    # _x = np.array([-2.5, -1.5, -0.5, -0.7, 0.5, 1.5])
    # import trading_data as tdata
    # # _x = tdata.DatasetGenerator.systhesis_input_generator(100)
    # _x = _x * 10
    # # _x = _x.reshape((1, -1, 1))
    # _x = _x.reshape((1, -1, 2))
    # x = ops.convert_to_tensor(_x, dtype=tf.float32)
    # LOG.debug("x.shape: {}".format(x.shape))
    # initial_state = None
    # layer = Operator(debug=True, weight=3)
    # y = layer(x, initial_state)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # y_res = sess.run(y)
    # # LOG.debug("y: {}".format(y_res))
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

    LOG.debug(colors.red("Test Operator"))
    fname = constants.FNAME_FORMAT["operators"].format(method="sin", weight=1, width=1)

    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    LOG.debug("timestap is: {}".format(inputs.shape[0]))

    batch_size = 20
    epochs = 5000 // batch_size
    steps_per_epoch = batch_size
    units = 10

    play = Play(batch_size=batch_size,
                units=units,
                activation=None,
                network_type=constants.NetworkType.OPERATOR)

    play.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)

    # LOG.debug("number of layer is: {}".format(play.number_of_layers))
    # LOG.debug("weight: {}".format(play.weight))

    LOG.debug(colors.red("Test Play"))
    batch_size = 10
    units = 20
    epochs = 1500 // batch_size
    steps_per_epoch = batch_size

    fname = constants.FNAME_FORMAT["plays"].format(method="sin", weight=1, width=1, points=500)

    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    length = 20
    inputs, outputs = inputs[:length], outputs[:length]

    # LOG.debug("timestap is: {}".format(inputs.shape[0]))

    # import time
    # start = time.time()
    # play = Play(batch_size=batch_size,
    #             units=units,
    #             activation="tanh",
    #             network_type=constants.NetworkType.PLAY,
    #             loss='mse',
    #             debug=True)

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

    batch_size = 10
    units = 20
    epochs = 100 // batch_size
    steps_per_epoch = batch_size
    mu = 0
    sigma = 0.01

    # fname = constants.FNAME_FORMAT["plays"].format(method="sin", weight=1, width=1, points=500)
    fname = constants.FNAME_FORMAT["F_predictions"].format(method="sin", weight=1, width=1, points=500, mu=mu, sigma=sigma, activation='tanh',
                                                           units=1, loss='mse')
    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    length = 50
    inputs, outputs = inputs[:length], outputs[:length]

    nb_plays = 2
    LOG.debug("timestap is: {}".format(inputs.shape[0]))
    import time
    start = time.time()
    agent = MyModel(batch_size=batch_size,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays)
    agent.fit2(inputs, mu, sigma, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # agent.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    # prediction, mu_prediction, sigma_prediction = agent.predict2(inputs)
    # prediction = agent.predict(inputs)
    point_ = prediction.shape[-1]
    fname = constants.FNAME_FORMAT["G_predictions"].format(method="sin", weight=1, width=1, points=point_, mu=mu_prediction, sigma=sigma_prediction, activation='tanh',
                                                           units=units, loss='mse')
    tdata.DatasetSaver.save_data(inputs, prediction, fname)

    LOG.debug("print weights info")
    agent.weights


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
