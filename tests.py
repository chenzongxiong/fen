import unittest
import core
import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


session = utils.get_session()


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


if __name__ == '__main__':
    unittest.main()
