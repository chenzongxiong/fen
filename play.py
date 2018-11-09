import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
import numpy as np

import utils
import constants
import log as logging


LOG = logging.getLogger(__name__)

sess = tf.Session()


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
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
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
        # TODO: test keras case
        # import ipdb; ipdb.set_trace()
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        state = ops.convert_to_tensor(state, dtype=self.dtype)

        outputs_ = tf.multiply(inputs, self.kernel)
        outputs = [state]

        if inputs.shape.ndims == 1:
            # simple tensorflow call, doesn't introduce `batch`
            for index in range(outputs_.shape[-1].value):
                phi_ = Phi(outputs_[index]-outputs[-1], width=self.width) + outputs[-1]
                outputs.append(phi_)
        elif inputs.shape.ndims == 2:
            # call from keras, introduce `batch` here
            for index in range(outputs_.shape[-1].value):
                phi_ = Phi(outputs_[:,index]-outputs[-1], width=self.width) + outputs[-1]
                outputs.append(phi_)
        else:
            raise

        outputs = tf.convert_to_tensor(outputs[1:])

        outputs = tf.reshape(outputs, shape=(-1, outputs.shape[0].value))
        # assert outputs.shape[-1] == inputs.shape[-1]

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

            self.state = tf.Variable(0,
                                     name="state",
                                     dtype=tf.float32)
            if constants.DEBUG_INIT_TF_VALUE:
                self.kernel1 = self.kernel1.initialized_value()
                self.kernel2 = self.kernel2.initialized_value()
                self.bias1 = self.bias1.initialized_value()
                self.bias2 = self.bias2.initialized_value()
                self.state = self.state.initialized_value()

            self._trainable_weights.append(self.kernel1)
            self._trainable_weights.append(self.kernel2)
            self._trainable_weights.append(self.bias1)
            self._trainable_weights.append(self.bias2)
            self._trainable_weights.append(self.state)

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

            self.state = self.add_weight(
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
        # import ipdb; ipdb.set_trace()
        # make it work under keras
        # if inputs.shape.ndims == 2:
        #      if inputs.shape[1].value == 1:
        #         inputs = tf.reshape(inputs, shape=(inputs.shape[0].value,))
        #     else:
        #         raise

        # if isinstance(inputs, np.ndarray):
        #     size_of_sequence = inputs.shape[0]
        # else:
        #     size_of_sequence = inputs.shape[0].value
        #     # size_of_sequence = inputs.shape[-1].value

        # size_per_chunk = size_of_sequence // self.nbr_of_chunks

        # print("Using chunks: ", self.nbr_of_chunks)

        # outputs1_list = []

        # for i in range(self.nbr_of_chunks):
        #     if i == 0:
        #         # question: only one weight or multiple weights?
        #         outputs1_ = self.cell(inputs[i*size_per_chunk:(i+1)*size_per_chunk], self.state)
        #     else:
        #         state = outputs1_list[-1][-1]   # retrieve the last output in previous play as intial state
        #         outputs1_ = self.cell(inputs[i*size_per_chunk:(i+1)*size_per_chunk], state)

        #     outputs1_list.append(outputs1_)

        # outputs1_ = tf.convert_to_tensor(outputs1_list)

        # outputs1_ = tf.reshape(outputs1_, shape=(outputs1_.shape[1].value * outputs1_.shape[0].value,))
        outputs1_ = self.cell(inputs, self.state)   # shape = (None, 1200)
        outputs1 = outputs1_ * self.kernel1         # shape = (4, 1200)
        assert outputs1.shape.ndims == 2

        if self.bias1 is not None:
            outputs1 += self.bias1
        if self.activation is not None:
            outputs1 =  self.activation(outputs1)

        # move forward
        outputs2 = outputs1 * self.kernel2         # shape = (4, 1200)
        outputs2 = tf.reduce_sum(outputs2, axis=0)   # shape = (1200, )

        if self.bias2 is not None:
            outputs2 += self.bias2

        assert outputs2.shape.ndims == 1
        # outputs2 = tf.reshape(outputs2, shape=(outputs2.shape[0], 1))
        return outputs2                   # shape = (1200,)

    def compute_output_shape(self, input_shape):
        # import ipdb; ipdb.set_trace()
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape.ndims == 1:
            output_shape = (input_shape[0].value, 1)
        elif input_shape.ndims == 2:
            if input_shape[1].value == 1:
                output_shape = (input_shape[0].value, 1)
            else:
                raise
        return tensor_shape.TensorShape(output_shape)

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


if __name__ == "__main__":
    method = "sin"
    weight = 1.0
    width = 1

    fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    inputs, outputs_ = utils.load(fname)
    outputs_ = outputs_.reshape((outputs_.shape[1],))

    # state = tf.constant(0, dtype=tf.float32)
    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)

    cell = PlayCell(weight=weight, width=width, debug=False)
    layer = Play(units=4, cell=cell,
                 activation="tanh",
                 debug=False)

    predictions = layer(inputs, state)

    loss = tf.losses.mean_squared_error(labels=outputs_,
                                        predictions=predictions)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # tensorboard writer
    # tensorboard --logdir .
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())

    sess.run(init)

    epochs = 1000
    loss_value = -1;

    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        LOG.debug("epoch", i, "loss:", loss_value,
                  ", weight: ", cell.kernel.eval(session=sess).tolist(),
                  ", theta1: ", layer.kernel1.eval(session=sess).tolist(),
                  ", theta2: ", layer.kernel2.eval(session=sess).tolist(),
                  ", bias1: ", layer.bias1.eval(session=sess).tolist(),
                  ", bias2: ", layer.bias2.eval(session=sess).tolist())

    LOG.debug("========================================")
