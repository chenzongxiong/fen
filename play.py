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


class PICell(Layer):
    def __init__(self,
                 units=1,
                 activation="tanh",
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 weight=1.0,
                 **kwargs):
        super(PICell, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)


        self.units = int(units)
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
            "units": self.units
        }
        base_config = super(PICell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":

    a = tf.constant([3.0, -0.5, -1, -2], dtype=tf.float32)

    pi_cell = PICell(units=1)
    sess.run(pi_cell(a))
