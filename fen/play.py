import numpy as np
import tensorflow as tf

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints


from fen import log as logging
from fen import colors
LOG = logging.getLogger(__name__)


def Phi(x, width=1.0):
    '''
    Phi(x) = x - width/2 , if x > width/2
           = x + width/2 , if x < - width/2
           = 0         , otherwise
    '''
    assert x.shape[0].value == 1 and x.shape[1].value == 1, "x must be a scalar"

    ZEROS = tf.zeros(x.shape, dtype=tf.float32, name='zeros')
    # _width = tf.constant([[width/2.0]], dtype=tf.float32)
    _width = tf.constant([[width/2.0]], dtype=tf.float32)
    r1 = tf.cond(tf.reduce_all(tf.less(x, -_width)), lambda: x + _width, lambda: ZEROS)
    r2 = tf.cond(tf.reduce_all(tf.greater(x, _width)), lambda: x - _width, lambda: r1)
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
        self.unroll = kwargs.pop('unittest', False)

        super(PhiCell, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self._weight = weight
        self._recurrent_weight = -1
        self._width = width
        self.units = 1
        self.state_size = [1]

        self.kernel_initializer = tf.keras.initializers.Constant(value=weight, dtype=tf.float32)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.hysteretic_func = hysteretic_func
        self.input_dim = input_dim

    def build(self, input_shape):
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
        LOG.debug("PhiCellinputs.shape: {}".format(inputs.shape))
        LOG.debug("PhiCell._inputs.shape: {}".format(self._inputs.shape))
        LOG.debug("PhiCell._state.shape: {}".format(self._state.shape))
        ############### IMPL from Scratch #####################
        # outputs_ = tf.multiply(self._inputs, self.kernel)
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
        def inner_steps(inputs, states):
            LOG.debug("inputs: {}, states: {}".format(inputs, states))
            outputs = self.hysteretic_func(inputs - states[-1], self._width) + states[-1]
            return outputs, [outputs]

        # import ipdb; ipdb.set_trace()
        self._inputs = tf.multiply(self._inputs, self.kernel)
        inputs_ = tf.reshape(self._inputs, shape=(1, self._inputs.shape[0].value*self._inputs.shape[1].value, 1))
        if isinstance(states, list) or isinstance(states, tuple):
            self._states = ops.convert_to_tensor(states[-1], dtype=tf.float32)
        else:
            self._states = ops.convert_to_tensor(states, dtype=tf.float32)

        assert self._state.shape.ndims == 2, colors.red("PhiCell states must be 2 dimensions")
        states_ = [tf.reshape(self._states, shape=self._states.shape.as_list())]
        last_outputs_, outputs_, states_x = tf.keras.backend.rnn(inner_steps, inputs=inputs_, initial_states=states_, unroll=self.unroll)

        LOG.debug("outputs_.shape: {}".format(outputs_))
        LOG.debug("states_x.shape: {}".format(states_x))
        return outputs_, list(states_x)


class Play(RNN):
    def __init__(self,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 unittest=False):

        cell = PhiCell(
            weight=weight,
            width=1.0,
            debug=debug,
            unittest=unittest)

        unroll = True if unittest is True else False
        super(Play, self).__init__(
            cell=cell,
            return_sequences=True,
            return_state=False,
            stateful=True,
            unroll=unroll)

    def call(self, inputs, initial_state=None):
        LOG.debug("Operator.inputs.shape: {}".format(inputs.shape))
        output = super(Play, self).call(inputs, initial_state=initial_state)
        assert inputs.shape.ndims == 3, colors.red("ERROR: Input from Operator must be 3 dimensions")
        shape = inputs.shape.as_list()
        output_ = tf.reshape(output, shape=(shape[0], -1, 1))
        return output_

    @property
    def kernel(self):
        return self.cell.kernel

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the batch size by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            self.states = [
                           tf.keras.backend.zeros([batch_size] + tensor_shape.as_shape(dim).as_list())
                           for dim in self.cell.state_size
                           ]
        elif states is None:
            # import ipdb; ipdb.set_trace()
            for state, dim in zip(self.states, self.cell.state_size):
                tf.keras.backend.set_value(state,
                                           np.zeros([batch_size] +
                                                    tensor_shape.as_shape(dim).as_list()))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' + str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                dim = self.cell.state_size[index]
                if value.shape != tuple([batch_size] +
                                        tensor_shape.as_shape(dim).as_list()):
                    raise ValueError(
                        'State ' + str(index) + ' is incompatible with layer ' +
                        self.name + ': expected shape=' + str(
                            (batch_size, dim)) + ', found shape=' + str(value.shape))
                # TODO(fchollet): consider batch calls to `set_value`.
                tf.keras.backend.set_value(state, value)
