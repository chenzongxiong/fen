import numpy as np
import tensorflow as tf

from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Dense

from fen import log as logging
from fen import colors

LOG = logging.getLogger(__name__)


class MyDense(Layer):
    def __init__(self, units=1,
                 activation="tanh",
                 weight=1,
                 use_bias=True,
                 activity_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        self._debug = kwargs.pop("debug", False)
        if '_init_kernel' in kwargs:
            self._init_kernel = kwargs.pop("_init_kernel")
        # self._init_kernel = 1.0
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
            outputs = self.activation(outputs)

        return outputs


class MySimpleDense(Dense):
    def __init__(self, **kwargs):
        self._debug = kwargs.pop("debug", False)
        self._init_bias = kwargs.pop("_init_bias", 0)
        if '_init_kernel' in kwargs:
            self._init_kernel = kwargs.pop("_init_kernel")
        # self._init_kernel = 1.0
        kwargs['activation'] = None
        kwargs['kernel_constraint'] = None
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
