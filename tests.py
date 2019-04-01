import unittest
import core
import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.parallel_for.gradients import jacobian
import pickle
import pickletools

session = utils.get_session(interactive=True)


class Foo():
    def __init__(self):
        self.foo = 0

    def bar(self, x):
        print("bar {}".format(x))

    def baz(self):
        print("baz")

    def __getstate__(self):
        # return self.__dict__
        return {'foo': 3, 'bax': 4, 'func': self.bar}
    # def __setstate(self, state):
    #     import ipdb; ipdb.set_trace()


class TestCases(unittest.TestCase):
    def setUp(self):
        self.inputs = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 1/3.0, 0.1,
                                0, 0.21, -1.5, 0.7, 0.9, 1.5, -0.4, 1, -0.15, 2])
        self.truth = np.array([0.5, 1, 2, 2, 0, 0, -0.5, -0.25, -1/6, -1/6,
                               -1/6, -1/6, -1, 0.2, 0.4, 1, 0.1, 0.5, 0.35, 1.5])

    def tearDown(self):
        pass

    def test_phi(self):
        a = tf.constant([[1]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))
        self.assertTrue(aa[0, 0] == 0.5)

        a = tf.constant([[0.5]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))
        self.assertTrue(aa[0, 0] == 0)

        a = tf.constant([[-0.5]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))
        self.assertTrue(aa[0, 0] == 0)

        a = tf.constant([[-1]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))
        self.assertTrue(aa[0, 0] == -0.5)

        a = tf.constant([[0]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))

        self.assertTrue(aa[0, 0] == 0)

        a = tf.constant([[100]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))
        self.assertTrue(aa[0, 0] == 99.5)

        a = tf.constant([[-100]], dtype=tf.float32, name="a")
        aa = session.run(core.Phi(a))
        self.assertTrue(aa.shape == (1, 1))
        self.assertTrue(aa[0, 0] == -99.5)

    def test_phicell(self):
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        cell = core.PhiCell(debug=True)
        output_1 = cell(input_1, [[[0]]])
        utils.init_tf_variables()
        input_2 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
        output_2 = cell(input_1, [[[0]]])
        utils.init_tf_variables()

        result_1, result_2 = session.run([output_1, output_2])
        # check outputs with different input sequence
        self.assertTrue(np.all(result_1[0].reshape(-1) == result_2[0][0].reshape(-1)))
        # check state with different input sequence
        self.assertTrue(result_1[1][0].reshape(-1)[0] == result_2[1][0].reshape(-1)[0])

        # check the value of outputs
        self.assertTrue(np.allclose(result_1[0].reshape(-1), self.truth, atol=1e-5))
        # check the value of state
        self.assertTrue(result_1[1][0].reshape(-1)[0] == 1.5)

    def test_operator(self):
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        operator_1 = core.Operator(debug=True)
        output_1 = operator_1(input_1)
        input_2 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
        operator_2 = core.Operator(debug=True)
        output_2 = operator_2(input_2)
        utils.init_tf_variables()
        result_1, result_2 = session.run([output_1, output_2])
        self.assertTrue(np.allclose(result_1.reshape(-1), self.truth))
        self.assertTrue(np.allclose(result_2.reshape(-1), self.truth))

    def test_mydense(self):
        units = 10
        _init_bias = 1
        _init_kernel = 2
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        mydense = core.MyDense(units=units, activation=None, use_bias=True, debug=True, _init_bias=_init_bias, _init_kernel=_init_kernel)

        output_1 = mydense(input_1)
        utils.init_tf_variables()
        result_1 = session.run(output_1)
        # check shape, must be equal to ()
        self.assertTrue(result_1.shape == (1, self.inputs.shape[0], units))
        # check value
        kernel = np.array([_init_kernel] * units).reshape(1, units)
        bias = np.array([_init_bias])
        truth = self.inputs.reshape([1, -1, 1]) * kernel + bias
        _truth = self.inputs * _init_kernel + _init_bias
        self.assertTrue(np.allclose(result_1, truth, atol=1e-5))
        self.assertTrue(np.allclose(_truth, truth[0, :, 0], atol=1e-5))

    def test_mysimpledense(self):
        units = 1
        _init_kernel = 2
        _init_bias = 1
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
        mysimpledense = core.MySimpleDense(units=units, _init_kernel=_init_kernel, _init_bias=_init_bias, use_bias=True, debug=True)
        output_1 = mysimpledense(input_1)
        utils.init_tf_variables()
        result_1 = session.run(output_1)
        kernel = np.array([_init_kernel]*2).reshape(-1, 1)
        bias = np.array([_init_bias])
        truth = np.matmul(self.inputs.reshape([1, -1, 2]), kernel) + bias
        self.assertTrue(np.allclose(result_1, truth, atol=1e-5))

    def test_average_layer(self):
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        average = tf.keras.layers.Average()([input_1, 3*input_1])
        utils.init_tf_variables()
        result = session.run(average)
        self.assertTrue(np.allclose(result.reshape(-1), 2*self.inputs.reshape(-1), atol=1e-5))

    def test_gradient_operator(self):
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        operator = core.Operator(debug=False)
        output_1 = operator(input_1)
        gradient_by_tf = tf.gradients(output_1, input_1)[0]
        gradient_by_hand = core.gradient_operator(output_1, operator.kernel)
        Jacobian = core.jacobian(output_1, input_1)
        utils.init_tf_variables()
        result_by_tf, result_by_hand, J = session.run([gradient_by_tf, gradient_by_hand, Jacobian])

        self.assertTrue(np.allclose(np.diag(J).reshape(-1), result_by_hand.reshape(-1)))
        self.assertTrue(np.allclose(J.sum(axis=0).reshape(-1), result_by_tf.reshape(-1)))

    def test_gradient_mydense(self):
        self._test_gradient_mydense_helper(activation=None)
        self._test_gradient_mydense_helper(activation='tanh')
        self._test_gradient_mydense_helper(activation='relu')

    def test_gradient_mysimpledense(self):
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
        mysimpledense = core.MySimpleDense(units=1,
                                           use_bias=True,
                                           debug=False)
        output_1 = mysimpledense(input_1)
        gradient_by_tf = tf.gradients(output_1, input_1)[0]
        utils.init_tf_variables()
        gradient_by_hand = core.gradient_linear_layer(mysimpledense.kernel,
                                                      multiples=self.inputs.shape[0]//input_1.shape[-1].value)

        result_by_tf, result_by_hand = session.run([gradient_by_tf, gradient_by_hand])
        # self.assertTrue(np.allclose(result_by_tf[:, 0].reshape(-1), result_by_hand.reshape(-1), atol=1e-5))
        self.assertTrue(np.allclose(result_by_tf, result_by_hand, atol=1e-5))

    def test_gradient_operator_mydense_None(self):
        self._test_gradient_operator_mydense_helper(activation=None)
        self._test_gradient_operator_mydense_helper(activation='tanh')
        self._test_gradient_operator_mydense_helper(activation='relu')

    def test_gradient_all(self):
        self._test_gradient_all_helper(activation=None)
        self._test_gradient_all_helper(activation='tanh')
        self._test_gradient_all_helper(activation='relu')
        self._test_gradient_all_helper(activation=None, input_dim=2)
        self._test_gradient_all_helper(activation='tanh', input_dim=2)
        self._test_gradient_all_helper(activation='relu', input_dim=2)
        self._test_gradient_all_helper(activation=None, input_dim=10)
        self._test_gradient_all_helper(activation='tanh', input_dim=10)
        self._test_gradient_all_helper(activation='relu', input_dim=10)

    def _test_gradient_mydense_helper(self, activation):
        units = 10
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        mydense = core.MyDense(units=units,
                               activation=activation,
                               use_bias=True,
                               debug=False)

        output_1 = mydense(input_1)
        gradient_by_tf = tf.gradients(output_1, input_1)[0]
        gradient_by_hand = core.gradient_nonlinear_layer(output_1, mydense.kernel, activation=activation)

        utils.init_tf_variables()
        result_by_tf, result_by_hand = session.run([gradient_by_tf, gradient_by_hand])
        self.assertTrue(np.allclose(result_by_tf.reshape(-1), result_by_hand.reshape(-1), atol=1e-5))

    def _test_gradient_operator_mydense_helper(self, activation):
        units = 5
        debug = False
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        operator = core.Operator(debug=debug)
        output_1 = operator(input_1)
        mydense = core.MyDense(units=units,
                               activation=activation,
                               use_bias=True,
                               debug=debug)
        output_2 = mydense(output_1)
        gradient_by_tf = tf.gradients(output_2, input_1)[0]

        gradient_by_hand = core.gradient_operator_nonlinear_layers(output_1,
                                                                   output_2,
                                                                   operator.kernel,
                                                                   mydense.kernel,
                                                                   activation,
                                                                   debug=True,
                                                                   inputs=input_1)

        utils.init_tf_variables()
        result_by_tf, result_by_hand = session.run([gradient_by_tf, gradient_by_hand])
        self.assertTrue(np.allclose(result_by_tf, result_by_hand))

    def _test_gradient_all_helper(self, activation, input_dim=1):
        units = 5
        debug = False
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, input_dim]), dtype=tf.float32)
        operator = core.Operator(debug=debug)
        output_1 = operator(input_1)
        mydense = core.MyDense(units=units,
                               activation=activation,
                               use_bias=True,
                               debug=debug)
        output_2 = mydense(output_1)

        mysimpledense = core.MySimpleDense(units=1,
                                           use_bias=True,
                                           activation=None,
                                           debug=debug)
        output_3 = mysimpledense(output_2)
        gradient_by_tf = tf.reshape(tf.gradients(output_3, input_1)[0], shape=output_1.shape)
        gradient_by_hand = core.gradient_all_layers(output_1,
                                                    output_2,
                                                    operator.kernel,
                                                    mydense.kernel,
                                                    mysimpledense.kernel,
                                                    activation,
                                                    debug=True,
                                                    inputs=input_1)
        utils.init_tf_variables()
        result_by_tf, result_by_hand = session.run([gradient_by_tf, gradient_by_hand])
        self.assertTrue(np.allclose(result_by_tf, result_by_hand))


    def test_multiple_plays(self):
        nb_plays = 2
        units = 5
        input_dim = 2
        activation = None
        timestep = self.inputs.shape[0] // input_dim
        mymodel = core.MyModel(nb_plays=nb_plays,
                               units=units,
                               input_dim=input_dim,
                               timestep=timestep,
                               activation=activation,
                               debug=True)
        mymodel.compile(self.inputs, mu=0, sigma=1, unittest=True)
        utils.init_tf_variables()
        result_by_tf, result_by_hand, result_J_list_by_tf, result_J_list_by_hand = session.run([mymodel.J_by_tf, mymodel.J_by_hand,
                                                                                                mymodel.J_list_by_tf, mymodel.J_list_by_hand], feed_dict=mymodel._x_feed_dict)
        for by_tf, by_hand in zip(result_J_list_by_tf, result_J_list_by_hand):
            if not np.allclose(by_hand, by_tf, atol=1e-5):
                print("ERROR: ")
                import ipdb; ipdb.set_trace()

        self.assertTrue(np.allclose(result_by_tf, result_by_hand, atol=1e-5))


    # def test_pickle(self):
    #     foo = Foo()
    #     p_foo = pickle.dumps(foo)
    #     print(pickletools.dis(p_foo))
    #     a = pickle.loads(p_foo)
    #     a.bar("world")


if __name__ == '__main__':
    unittest.main()
