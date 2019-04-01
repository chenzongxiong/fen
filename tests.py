import unittest
import core
import utils
import numpy as np
import tensorflow as tf


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




if __name__ == '__main__':
    unittest.main()
