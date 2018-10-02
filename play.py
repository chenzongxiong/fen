import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers

sess = tf.Session()


def Phi(x, width=1):
    """
    Phi(x) = x         , if x > 0
           = x + width , if x < - width
           = 0         , otherwise
    """
    return tf.math.maximum(0.0, x) + tf.math.minimum(0.0, x+width)


class PlayCell(Layer):
    def __init__(self,
                 # units=1,
                 # activation="tanh",
                 # use_bias=False,
                 # kernel_initializer='glorot_uniform',
                 # bias_initializer='zeros',
                 # kernel_regularizer=None,
                 # bias_regularizer=None,
                 # activity_regularizer=None,
                 # kernel_constraint=None,
                 # bias_constraint=None,
                 weight=1.0,
                 **kwargs):
        # super(PlayCell, self).__init__(
        #     activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        super(PlayCell, self).__init__(**kwargs)
        # self.units = int(units)
        self.units = 1
        self.weight = weight

    def build(self, input_shape):
        self.kernel = tf.Variable(self.weight, "kernel").initialized_value()
        self._trainable_weights.append(self.kernel)

        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        p0 = inputs[0]
        outputs_ = tf.multiply(inputs[1:], self.kernel)

        outputs = [p0]
        for index in range(outputs_.shape[-1].value):
            outputs.append(Phi(outputs_[index] - outputs[-1]) + outputs[-1])
        outputs = tf.convert_to_tensor(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        return input_shape

    def get_config(self):
        config = {
            "units": self.units,
            "weight": self.weight
        }
        base_config = super(PlayCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PILayer(Layer):
    def __init__(self,
                 units,
                 players,
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

        super(PILayer, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units)
        # players: a list of play cell
        self.players = players
        self.number_of_players = len(players)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        # inputs should be like p0, x1, x2, x3, ..., xm
        # inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        outputs_list = []
        for player in self.players:
            outputs_list.append(player.__call__(inputs))

        outputs_ = ops.convert_to_tensor(outputs_list, dtype=self.dtype)
        outputs = tf.mat_mul(outputs_, self.kernel)

        # if self.use_bias:
        #     outputs = nn.bias_add(outputs, self.bias)
        # if self.activation is not None:
        #     return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass



if __name__ == "__main__":
    a = tf.constant([3.0, -0.5, -1, -2], dtype=tf.float32)
    pi_cell = PlayCell()
    print(sess.run(pi_cell(a)))
