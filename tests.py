import unittest
import numpy as np
import tensorflow as tf
from play import Phi


sess = tf.Session()


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


class TensorflowOpsTestCase(BaseTestCase):
    def setUp(self):
        self.sess = tf.Session()
        pass

    def test(self):
        from tensorflow.python.ops import gen_math_ops
        from tensorflow.python.framework import ops
        # inputs = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        inputs = tf.Variable([1.0, 2.0, 3.0, 4.0]).initialized_value()
        # inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)
        kernel = tf.Variable(2.0).initialized_value()
        # outputs = inputs * kernel
        print(self.sess.run(inputs[0]))
        outputs = tf.multiply(inputs[1:], kernel)
        # outputs = tf.matmul(inputs, kernel)
        # outputs = gen_math_ops.mat_mul(inputs, kernel)
        print(self.sess.run(outputs))
        print(inputs.shape[-1].value)
        outputs = []
        for index in range(inputs.shape[-1].value):
            outputs.append(inputs[index]-kernel)
            # assign = inputs[index].assign(inputs[index] - 1)
            # assign.op.run()
        outputs = tf.convert_to_tensor(outputs)
        print(outputs)
        print(self.sess.run(outputs))
        # for v in inputs:
        #     print(sess.run(v))
        # self.assertEqual()



class PlayTestCase(BaseTestCase):

    def test_phi(self):
        a = tf.constant([3.0, -0.5, -1., -2.], dtype=tf.float32)
        b = tf.constant([3, 0, 0, -1], dtype=tf.float32)
        a_ = Phi(a)
        result = tf.equal(a_, b)
        self.assertTrue(np.all(result.eval(session=sess)))


if __name__ == "__main__":
    unittest.main()
