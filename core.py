import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import RNN

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
import numpy as np

import utils
import colors
import constants

import log as logging


LOG = logging.getLogger(__name__)

sess = utils.get_session()


def Phi(x, width=1.0):
    """
    Phi(x) = x         , if x > 0
           = x + width , if x < - width
           = 0         , otherwise
    """
    return tf.maximum(x, 0) + tf.minimum(x+width, 0)


class PlayCell(Layer):
    def __init__(self,
                 weight=1.0,
                 width=1.0,
                 hysteretic_func=Phi,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 # kernel_constraint=None,
                 kernel_constraint="non_neg",
                 **kwargs):

        self.debug = kwargs.pop("debug", False)

        super(PlayCell, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.weight = weight
        self.width = width
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # TODO: try non negative constraint
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.hysteretic_func = hysteretic_func

    def build(self, input_shape):
        if self.debug:
            LOG.debug("Initialize *weight* as pre-defined...")
            self.kernel = tf.Variable(self.weight, name="kernel", dtype=tf.float32)
            if constants.DEBUG_INIT_TF_VALUE:
                self.kernel = self.kernel.initialized_value()

            self._trainable_weights.append(self.kernel)
        else:
            LOG.debug("Initialize *weight* randomly...")
            self.kernel = self.add_weight(
                'kernel',
                shape=(),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

        self.built = True

    def call(self, inputs, state):
        """
        Parameters:
        ----------------
        inputs: `inputs` is a vector
        state: `state` is randomly initialized
        """

        # inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        # if inputs.shape.ndims == 1:
        #     inputs = tf.reshape(inputs, shape=(1, -1))
        # elif inputs.shape.ndims > 2:
        #     raise Exception("len(inputs.shape) must be less or equal than 2, but got {}".format(inputs.shape.ndims))
        # outputs_ = tf.multiply(inputs, self.kernel)
        # outputs = [state]

        self._inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        self._state = ops.convert_to_tensor(state, dtype=self.dtype)
        if self._inputs.shape.ndims == 1:
            self._inputs = tf.reshape(self._inputs, shape=(1, -1))
        elif self._inputs.shape.ndims > 2:
            raise Exception("len(inputs.shape) must be less or equal than 2, but got {}".format(self._inputs.shape.ndims))

        outputs_ = tf.multiply(self._inputs, self.kernel)
        # outputs = [self._state]

        # for index in range(outputs_.shape[-1].value):
        #     phi_ = self.hysteretic_func(outputs_[:, index]-outputs[-1], width=self.width) + outputs[-1]
        #     outputs.append(phi_)

        # outputs = tf.convert_to_tensor(outputs[1:])
        outputs = tf.convert_to_tensor(outputs_)

        outputs = tf.reshape(outputs, shape=(-1, outputs.shape[0].value))
        # LOG.debug("{} inputs.shape: {}, output.shape: {}".format(colors.red("PlayCell"),
        #                                                          inputs.shape, outputs.shape))

        # LOG.debug("{} inputs.shape: {}, output.shape: {}".format(colors.red("PlayCell"),
        #                                                          self._inputs.shape, outputs.shape))

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        return input_shape

    def get_config(self):
        config = {
            "debug": self.debug,
            "weight": self.weight,
            "width": self.width,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularize),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
        }
        base_config = super(PlayCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Play(Layer):
    def __init__(self,
                 units,
                 cell,
                 nbr_of_chunks=1,
                 activation="tanh",
                 use_bias=True,
                 fixed_state=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.debug = kwargs.pop("debug", False)

        super(Play, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units)
        self.cell = cell
        self.nbr_of_chunks = nbr_of_chunks

        if self.debug:
            self.units = 4

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        if self.debug:
            LOG.debug("Initalize *theta* as pre-defined...")
            self.kernel1 = tf.Variable([[1],
                                       [2],
                                       [3],
                                       [4]],
                                      name="theta1",
                                      dtype=tf.float32)

            self.bias1 = tf.Variable([[1],
                                      [2],
                                      [-1],
                                      [-2]],
                                    name="bias1",
                                    dtype=tf.float32)

            self.kernel2 = tf.Variable([[1],
                                        [2],
                                        [3],
                                        [4]],
                                       name="theta2",
                                       dtype=tf.float32)

            self.bias2 = tf.Variable(1,
                                     name="bias2",
                                     dtype=tf.float32)

            self._state = tf.Variable(0,
                                     name="state",
                                     dtype=tf.float32)
            if constants.DEBUG_INIT_TF_VALUE:
                self.kernel1 = self.kernel1.initialized_value()
                self.kernel2 = self.kernel2.initialized_value()
                self.bias1 = self.bias1.initialized_value()
                self.bias2 = self.bias2.initialized_value()
                self._state = self._state.initialized_value()

            self._trainable_weights.append(self.kernel1)
            self._trainable_weights.append(self.kernel2)
            self._trainable_weights.append(self.bias1)
            self._trainable_weights.append(self.bias2)
            self._trainable_weights.append(self._state)

        else:
            LOG.debug("Initalize *theta* randomly...")
            self.kernel1 = self.add_weight(
                'theta1',
                shape=(self.units, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            self.kernel2 = self.add_weight(
                'theta2',
                shape=(self.units, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            self._state = self.add_weight(
                'state',
                # shape=(self.nbr_of_cells, 1),
                shape=(),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            if self.use_bias:
                self.bias1 = self.add_weight(
                    'bias1',
                    shape=(self.units, 1),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    dtype=self.dtype,
                    trainable=True)

                self.bias2 = self.add_weight(
                    'bias2',
                    shape=(),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    dtype=self.dtype,
                    trainable=True)
            else:
                self.bias1 = None
                self.bias2 = None

        self.built = True

    def call(self, inputs):
        self._inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        outputs1_ = self.cell(self._inputs, self._state)
        # outputs1_ = self.cell(inputs, self._state)
        outputs1 = outputs1_ * self.kernel1
        assert outputs1.shape.ndims == 2

        if self.bias1 is not None:
            outputs1 += self.bias1
        if self.activation is not None:
            outputs1 =  self.activation(outputs1)

        # move forward
        outputs2 = outputs1 * self.kernel2
        outputs2 = tf.reduce_sum(outputs2, axis=0)

        if self.bias2 is not None:
            outputs2 += self.bias2

        # LOG.debug("{}, inputs.shape: {}, outputs.shape: {}".format(colors.red("Play"),
        #                                                            inputs.shape, outputs2.shape))

        return outputs2

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(input_shape)

    def get_config(self):
        config = {
            "debug": self.debug,
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(Play, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class PlayModel(tf.keras.Model):
    def __init__(self, nb_plays, units=4, batch_size=1, *args, **kwargs):
        super(PlayModel, self).__init__(name="play_model")

        self.debug = kwargs.pop("debug", False)

        self._nb_plays = nb_plays
        self._plays = []
        # self._batch_size = batch_size

        for _ in range(self._nb_plays):
            cell = PlayCell(debug=self.debug)
            play = Play(units=units, cell=cell, debug=self.debug)
            self._plays.append(play)
        self.plays_outputs = None

    def call(self, inputs, debug=False):
        """
        Parameters:
        ----------------
        inputs: `inputs` is a vector, assert len(inputs.shape) == 1
        """
        outputs = []
        self._inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        # inputs = self._inputs
        for play in self._plays:
            outputs.append(play(self._inputs))
            # outputs.append(play(inputs))

        outputs = tf.convert_to_tensor(outputs, dtype=self.dtype)
        self.plays_outputs = outputs
        # LOG.debug("{} outputs.shape: {}".format(colors.red("PlayModel"),
        #                                         outputs.shape))
        outputs = tf.reduce_sum(outputs, axis=0)
        # LOG.debug("{} outputs.shape: {}".format(colors.red("PlayModel"),
        #                                         outputs.shape))
        outputs = tf.reshape(outputs, shape=(-1, outputs.shape[0].value))
        if debug is True:
            return outputs, self.plays_outputs
        else:
            return outputs

    def get_config(self):
        config = {
            "nb_plays": self._nb_plays,
            "debug": self.debug,
        }

        return config

    def get_plays_outputs(self, inputs, batch_size=1):
        assert len(inputs.shape) == 2
        sess = utils.get_session()
        samples, _ = inputs.shape
        plays_outputs_list = []
        for x in range(samples):
            outputs, plays_outputs = self.__call__(inputs[x,:], debug=True)
            outputs_eval = sess.run(outputs)
            plays_outputs_eval = sess.run(plays_outputs)
            plays_outputs_list.append(plays_outputs_eval)

        return np.hstack(plays_outputs_list).T


class PlayModel2(Layer):
    def __init__(self, nb_plays, units=4, batch_size=1,
                 activity_regularizer=None,
                 **kwargs):
        self.debug = kwargs.pop("debug", False)
        super(PlayModel2, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self._nb_plays = nb_plays
        self._plays = []
        for _ in range(self._nb_plays):
            cell = PlayCell(debug=self.debug)
            play = Play(units=units, cell=cell, debug=self.debug)
            self._plays.append(play)
        self.plays_outputs = None

    def call(self, inputs, debug=False):
        """
        Parameters:
        ----------------
        inputs: `inputs` is a vector, assert len(inputs.shape) == 1
        """
        outputs = []
        self._inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)

        for play in self._plays:
            outputs.append(play(self._inputs))

        outputs = tf.convert_to_tensor(outputs, dtype=self.dtype)
        # LOG.debug("{} outputs.shape: {}".format(colors.red("PlayModel"),
        #                                         outputs.shape))
        outputs = tf.reduce_sum(outputs, axis=0)
        # LOG.debug("{} outputs.shape: {}".format(colors.red("PlayModel"),
        #                                         outputs.shape))
        outputs = tf.reshape(outputs, shape=(-1, outputs.shape[0].value))

        return outputs


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
            LOG.debug("Initialize *weight* as pre-defined...")
            self.kernel = tf.Variable([[self._weight]], name="weight", dtype=tf.float32)
            if constants.DEBUG_INIT_TF_VALUE:
                self.weight = self.kernel.initialized_value()

            self._trainable_weights.append(self.weight)
        else:
            LOG.debug("Initialize *weight* randomly...")
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
        # unroll method, can we use RNN method ?
        outputs_ = tf.multiply(self._inputs, self.kernel)
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
            stateful=True
            )

    def call(self, inputs, inital_state=None):
        return super(Operator, self).call(
            inputs, initial_state=initial_state
        )


if __name__ == "__main__":
    # set random seed to make results reproducible
    tf.random.set_random_seed(123)
    np.random.seed(123)

    sess = utils.get_session()
    phi_cell = PhiCell(weight=1.0, width=1.0, debug=True)
    _x = np.array([-2.5, -1.5, -0.5, -0.7, 0.5, 1.5])
    outputs = phi_cell(_x, [0])

    init = tf.global_variables_initializer()
    sess.run(init)

    LOG.debug("outputs: {}".format(sess.run(outputs)))

    # _x = np.array([-2.5, -1.5, -0.5, -0.7, 0.5, 1.5])
    import trading_data as tdata
    _x = tdata.DatasetGenerator.systhesis_input_generator(100)
    _x = _x * 10
    _x = _x.reshape((1, -1, 1))
    # _x = _x.reshape((1, -1, 2))
    x = ops.convert_to_tensor(_x, dtype=tf.float32)
    LOG.debug("x.shape: {}".format(x.shape))
    initial_state = None
    layer = Operator(debug=True, weight=3)
    y = layer(x, initial_state)
    init = tf.global_variables_initializer()
    sess.run(init)
    y_res = sess.run(y)
    # LOG.debug("y: {}".format(y_res))
    _y_true = y_res

    import time
    start = time.time()
    model = tf.keras.models.Sequential()

    _x = _x.reshape((1, -1, 10))
    _y_true = _y_true.reshape((1, -1, 10))
    x = ops.convert_to_tensor(_x, dtype=tf.float32)
    y_true = ops.convert_to_tensor(_y_true, dtype=tf.float32)

    model.add(tf.keras.layers.InputLayer(batch_size=1, input_shape=x.shape[1:]))
    model.add(Operator())
    model.compile(loss="mse",
                  optimizer="adam",
                  metrics=["mse"])

    model.fit(x, y_true, epochs=2500, batch_size=None, verbose=1, steps_per_epoch=1, shuffle=False)
    end = time.time()
    LOG.debug("time costs: {} s".format(end-start))
    score = model.evaluate(x, y_true, steps=1)
    LOG.debug("score: {}".format(score))
    LOG.debug("weight: {}".format(sess.run(model._layers[1].cell.kernel)))
