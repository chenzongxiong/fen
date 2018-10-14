import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
import numpy as np


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

        self.units = 1
        self.weight = weight
        self.width = width
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        if self.debug:
            print("Initialize *weight* as pre-defined...")
            self.kernel = tf.Variable(self.weight, name="kernel", dtype=tf.float32).initialized_value()
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
            "units": self.units,
            "weight": self.weight,
            "width": self.width,
        }
        base_config = super(PlayCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PILayer(Layer):
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

        super(PILayer, self).__init__(
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
            self.kernel = tf.Variable([[1],
                                       [2],
                                       [3],
                                       [4]],
                                      name="kernel",
                                      dtype=tf.float32).initialized_value()
            self.bias = tf.Variable([[1], [2], [-1], [-2]],
                                    name="bias",
                                    # shape=(self.units, 1),
                                    dtype=tf.float32).initialized_value()
            self._trainable_weights.append(self.kernel)
            self._trainable_weights.append(self.bias)
        else:
            print("Initalize *theta* randomly...")
            self.kernel = self.add_weight(
                'kernel',
                shape=(self.units, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            if self.use_bias:
                self.bias = self.add_weight(
                    'bias',
                    shape=(self.units, 1),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    dtype=self.dtype,
                    trainable=True)
            else:
                self.bias = None
        self.built = True

    def call(self, inputs, state):
        # import ipdb; ipdb.set_trace()
        outputs_ = self.cell.__call__(inputs, state)

        outputs = outputs_ * self.kernel
        assert outputs.shape.ndims == 2
        # outputs = tf.reshape(outputs, shape=(outputs.shape[1].value, outputs.shape[0].value))
        # assert outputs.shape[1].value == self.units

        if self.use_bias:
            outputs += self.bias
        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # output_shape = [input_shape[-1].value, self.units]
        outputs_shape = (self.units, input_shape[-1].value)
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        pass


class PlayBlock(Layer):
    def __init__(self,
                 units,
                 number_of_players,
                 activation="tanh",
                 **kwargs):
        self.debug = kwargs.pop("debug", False)
        super(PlayBlock, self).__init__(**kwargs)

        if self.debug:
            units = 2
            self.units = units
            self.pi_layer = PILayer(units=units,
                                    number_of_players=number_of_players,
                                    debug=True)
        else:
            self.units = units
            self.pi_layer = PILayer(units=units,
                                    number_of_players=number_of_players)

        self.activation = activations.get(activation)

    def build(self, input_shape):
        if self.debug:
            self.kernel = tf.Variable([1, -1],
                                      name="kernel",
                                      trainable=True,
                                      dtype=tf.float32).initialized_value()
            self.bias = tf.Variable([1, 2],
                                    name="bias",
                                    dtype=tf.float32).initialized_value()
            self._trainable_weights.append(self.kernel)
            self._trainable_weights.append(self.bias)
        else:
            self.kernel = None
            self.bias = None

        self.built = True

    def call(self, inputs):
        outputs_ = self.pi_layer.__call__(inputs)
        outputs = outputs_ * self.kernel + self.bias
        outputs = tf.reduce_sum(outputs, axis=1)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        output_shape = input_shape[-1].value
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        pass


if __name__ == "__main__":
    # a = tf.constant([[1, 2], [2, 3], [3, 4]], dtype=tf.float32)
    # b = tf.constant([1, 3], dtype=tf.float32)
    # c = a*b+b
    # print(sess.run(tf.reduce_sum(c, axis=1)))
    # _inputs = np.random.randint(low=-10, high=10, size=500)
    _inputs = np.random.uniform(low=-10, high=10, size=5000)
    np.insert(_inputs, 0, 0)
    inputs = tf.constant(_inputs, dtype=tf.float32)

    pi_cell = PlayCell(weight=1.0)
    outputs = pi_cell(inputs)
    inputs_res = sess.run(inputs)
    outputs_res = sess.run(outputs)
    print(inputs_res, outputs_res)
    # plt.plot(inputs_res, outputs_res)
    plt.scatter(inputs_res, outputs_res)
    plt.show()
    # import ipdb; ipdb.set_trace()
    # print(sess.run(pi_cell(a)))

    # x = tf.constant([[-1, -2, -0.5, 1],
    #                  [0, -2, -0.5, 1],
    #                  [1, -2, -0.5, 1],
    #                  [10, -2, -0.5, 1]], dtype=tf.float32, shape=[4, 4])
    # y = tf.constant([1, 2, 3, 4], dtype=tf.float32, shape=(4,))
    # # print(sess.run(x + y))
    # # array([[-1. , -1. , -0.5,  1. ],
    # #        [ 0. , -3. , -1. ,  2. ],
    # #        [ 1. , -5. , -1.5,  3. ],
    # #        [10. , -7. , -2. ,  4. ]], dtype=float32)
    # pi_layer = PILayer(units=2, number_of_players=4, debug=True)
    # sess.run(pi_layer(x))
    # # array([[-1.5, -2. ],
    # #        [-2. , -4. ],
    # #        [-2.5, -6. ],
    # #        [ 5. ,  0. ]], dtype=float32)
