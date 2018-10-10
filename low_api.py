import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

sess = tf.Session()

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b
total2 = a + b
total3 = a + b
print(a)
print(b)
print(total)
print(total2)
print(total3)

a_result = sess.run(a)
b_result = sess.run(b)
total_result = sess.run(total)
print("a result: ", a_result)
print("b result: ", b_result)
print("total result: ", total_result)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
print("x: ", x)
print("y: ", y)
print("z: ", z)
z_result_by_variable = sess.run(z, feed_dict={x: 3, y: 4})
z_result_by_name = sess.run("add_3:0", feed_dict={"Placeholder:0": 3, "Placeholder_1:0": 5})
print("z result 1: ", z_result_by_variable)
print("z result 2: ", z_result_by_name)


from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import common_shapes
from tensorflow.python.keras.engine.base_layer import InputSpec

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.util.tf_export import tf_export


class MyDense(Layer):
    def __init__(self,
                units,
                activation=None,
                use_bias=False,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MyDense, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        # import ipdb; ipdb.set_trace()
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        # self.input_spec = InputSpec(min_ndim=1)

    def build(self, input_shape):
        # import ipdb; ipdb.set_trace()
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `MyDense` '
                             'should be defined. Found `None`.')
        # self.input_spec = InputSpec(min_ndim=2,
        #                             axes={-1: input_shape[-1].value})
        self.kernel = self.add_weight(
            'kernel',
            # shape=[input_shape[-1].value, self.units],
            shape=(),
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
        import ipdb; ipdb.set_trace()
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            # outputs = gen_math_ops.mat_mul(inputs, self.kernel)
            outputs = tf.multiply(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import ipdb; ipdb.set_trace()

# x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
# y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
y_true = tf.constant([0, -1, -2, -3], dtype=tf.float32)

linear_model = MyDense(units=1)
print("config: ", linear_model.get_config())
y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# tensorboard writer
# tensorboard --logdir .
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())


sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)


print(sess.run(y_pred))
# import ipdb; ipdb.set_trace()
for v in linear_model._trainable_weights:
    print(sess.run(v))


floating = tf.Variable(3.14, tf.float64)
rank = tf.rank(floating)
print("rank: ", rank)
# print("floating: ", floating.eval())
constant = tf.constant([1, 2, 3])
tensor = constant * constant
print("constant eval: ", constant.eval(session=sess))
print("tensor: ", tensor)
print("tensor eval: ", tensor.eval(session=sess))
print("========================================")

test = tf.Print(tensor, [constant, tensor])
result = test + 1
print("result eval: ", result.eval(session=sess))
