import sys
import argparse
import numpy as np
import tensorflow as tf

from core import Phi, PlayCell
import utils
import trading_data as tdata
import log as logging
import constants


writer = utils.get_tf_summary_writer("./log/Gops")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
points = 100

def fit(inputs, outputs, width, method, true_weight):
    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)
    cell = PlayCell(width=width,
                    kernel_constraint="non_neg",
                    debug=False)

    predictions = cell(inputs, state)
    loss = tf.losses.mean_squared_error(labels=outputs,
                                        predictions=predictions)

    loss_summary = tf.summary.scalar("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_value = None
    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        if i % 10 == 0:         # every 10 times
            summary = sess.run(loss_summary)
            writer.add_summary(summary, i)

        LOG.debug("epoch {}, loss: {}".format(i, loss_value))

    state = tf.constant(0, dtype=tf.float32)
    predictions = cell(inputs, state)
    return sess.run(predictions), float(loss_value)


if __name__ == '__main__':
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    # train dataset
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["Gops"].format(method=method, weight=weight, width=width, points=points)
                inputs, outputs = tdata.DatasetLoader.load_data(fname)
                inputs = inputs.reshape(1, -1)
                outputs = outputs.reshape(1, -1)
                predictions, loss = fit(inputs, outputs, width, method, weight)
                fname = constants.FNAME_FORMAT["Gops_loss"].format(method=method, weight=weight, width=width, points=points)
                tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                fname = constants.FNAME_FORMAT["Gops_predictions"].format(method=method, weight=weight, width=width, points=points)
                # hotfix:
                inputs = inputs.reshape(-1)
                predictions = predictions.T
                tdata.DatasetSaver.save_data(inputs, predictions, fname)
