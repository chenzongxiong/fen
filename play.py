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
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        if self.debug:
            print("Initialize *weight* as pre-defined...")
            # self.kernel = tf.Variable(self.weight, name="kernel", dtype=tf.float32).initialized_value()
            self.kernel = tf.Variable(self.weight, name="kernel", dtype=tf.float32)
            self._trainable_weights.append(self.kernel)
        else:
            print("Initialize *weight* randomly...")
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
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        state = ops.convert_to_tensor(state, dtype=self.dtype)

        outputs_ = tf.multiply(inputs, self.kernel)
        outputs = [state]

        for index in range(outputs_.shape[-1].value):
            phi_ = Phi(outputs_[index]-outputs[-1], width=self.width) + outputs[-1]
            outputs.append(phi_)

        outputs = tf.convert_to_tensor(outputs[1:])
        if outputs.shape.ndims == 2 and outputs.shape[1].value == 1:
            outputs = tf.reshape(outputs, shape=(outputs.shape[0].value,))
        elif outputs.shape.ndims > 2:
            raise

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        return input_shape

    def get_config(self):
        config = {
            "weight": self.weight,
            "width": self.width,
            "debug": self.debug,
        }
        base_config = super(PlayCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Play(Layer):
    def __init__(self,
                 units,
                 cell,
                 activation="tanh",
                 use_bias=True,
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
            print("Initalize *theta* as pre-defined...")
            self.kernel1 = tf.Variable([[1],
                                       [2],
                                       [3],
                                       [4]],
                                      name="kernel1",
                                      dtype=tf.float32)
            self.bias1 = tf.Variable([[1], [2], [-1], [-2]],
                                    name="bias1",
                                    # shape=(self.units, 1),
                                    dtype=tf.float32)

            self.kernel2 = tf.Variable([[1], [2], [3], [4]],
                                       name="kernel2",
                                       dtype=tf.float32)
            self.bias2 = tf.Variable(1,
                                     name="bias2",
                                     dtype=tf.float32)

            self._trainable_weights.append(self.kernel1)
            self._trainable_weights.append(self.kernel2)
            self._trainable_weights.append(self.bias1)
            self._trainable_weights.append(self.bias2)
        else:
            print("Initalize *theta* randomly...")
            self.kernel1 = self.add_weight(
                'kernel1',
                shape=(self.units, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            self.kernel2 = self.add_weight(
                'kernel2',
                shape=(self.units, 1),
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

    def call(self, inputs, state):

        outputs1_ = self.cell.__call__(inputs, state)
        outputs1 = outputs1_ * self.kernel1
        assert outputs1.shape.ndims == 2
        # outputs = tf.reshape(outputs, shape=(outputs.shape[1].value, outputs.shape[0].value))
        # assert outputs.shape[1].value == self.units

        if self.use_bias:
            outputs1 += self.bias1
        if self.activation is not None:
            outputs1 =  self.activation(outputs1)

        # move forward
        outputs2 = outputs1 * self.kernel2
        outputs2 = tf.reduce_sum(outputs2, axis=0)

        if self.use_bias:
            outputs2 += self.bias2
        if self.activation is not None:
            outputs2 = self.activation(outputs2)

        assert outputs2.shape.ndims == 1
        return outputs2

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        outputs_shape = (self.units, input_shape[-1].value)
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            "units": self.units,
            "debug": self.debug,
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
                 activation="linear",
                 debug=False)

    predictions = layer(inputs, state)

    loss = tf.losses.mean_squared_error(labels=outputs_,
                                        predictions=predictions)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    epochs = 1000
    loss_value = -1;

    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        print("epoch", i, "loss:", loss_value,
              ", weight: ", cell.kernel.eval(session=sess).tolist(),
              ", theta1: ", layer.kernel1.eval(session=sess).tolist(),
              ", theta2: ", layer.kernel2.eval(session=sess).tolist(),
              ", bias1: ", layer.bias1.eval(session=sess).tolist(),
              ", bias2: ", layer.bias2.eval(session=sess).tolist())

    print("========================================")
