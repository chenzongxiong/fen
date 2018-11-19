import tensorflow as tf

import losses
import utils
import log as logging


LOG = logging.getLogger(__name__)
sess = utils.get_session()


if __name__ == "__main__":
    B = tf.constant([1, 2, 10, 100], dtype=tf.float32)
    mu = tf.constant(5.0)
    tau = tf.constant(2.0)

    B1 = B[:-1]
    B2 = B[1:]
    B3 = tf.math.square(B[1:] - B[:-1] - mu)
    res = tf.reduce_sum(B3)
    LOG.debug(sess.run(B1))
    LOG.debug(sess.run(B2))
    LOG.debug(sess.run(res))

    # LOG.debug(sess.run(res - mu))
