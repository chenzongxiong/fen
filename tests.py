import unittest
import core
import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


session = utils.get_session()


class TestCases(unittest.TestCase):

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
        inputs = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 1/3.0, 0.1,
                           0, 0.21, -1.5, 0.7, 0.9, 1.5, -0.4, 1, -0.15, 2])
        input_1 = ops.convert_to_tensor(inputs.reshape([1, -1, 1]), dtype=tf.float32)
        cell = core.PhiCell(debug=True)
        output_1 = cell(input_1, [[[0]]])
        utils.init_tf_variables()
        input_2 = ops.convert_to_tensor(inputs.reshape([1, -1, 2]), dtype=tf.float32)
        output_2 = cell(input_1, [[[0]]])
        utils.init_tf_variables()

        result_1, result_2 = session.run([output_1, output_2])
        # check outputs with different input sequence
        self.assertTrue(np.all(result_1[0].reshape(-1) == result_2[0][0].reshape(-1)))
        # check state with different input sequence
        self.assertTrue(result_1[1][0].reshape(-1)[0] == result_2[1][0].reshape(-1)[0])

        # check the value of outputs
        truth = np.array([0.5, 1, 2, 2, 0, 0, -0.5, -0.25, -1/6, -1/6,
                          -1/6, -1/6, -1, 0.2, 0.4, 1, 0.1, 0.5, 0.35, 1.5])
        self.assertTrue(np.allclose(result_1[0].reshape(-1), truth, atol=1e-5))
        # check the value of state
        self.assertTrue(result_1[1][0].reshape(-1)[0] == 1.5)


if __name__ == '__main__':
    unittest.main()
