import tensorflow as tf


def mle_loss(B, W, mu, tau):
    """
    Calculate maximize log-likelihood
    Parameters:
    --------
    B: N by 1 vector. dtype: tf.float32
    W: N by N matrix (?). dtype: tf.float32
    mu: a scalar. dtype: tf.float32
    tau: a scalar, same as sigma in gaussian distribution. dtype: tf.float32
    """
    J = tf.gradient(B, [W])
    _B = tf.math.squre(B[1:] - B[:-1] - mu) - tf.math.log(tau)
    # TODO: multiply by Jaccobia matrix, unclear
    log_prob = _B * tf.linalg.det(J)

    neg_log_likelihood = tf.reduce_sum(log_prob)

    return neg_log_likelihood


def mle_gradient(loss, W, mu, tau, P0):
    return tf.gradient(loss, [W, mu, tau, P0])
