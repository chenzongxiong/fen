# import sys
# import argparse
import numpy as np
import tensorflow as tf

import utils
from core import Play, PlayCell
import log as logging
import constants
import trading_data as tdata

writer = utils.get_tf_summary_writer("./log/plays")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS


def fit(inputs, outputs, units, activation, width, true_weight):

    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)

    cell = PlayCell(weight=weight, width=width,
                    kernel_constraint="non_neg",
                    debug=False)

    layer = Play(units=units, cell=cell,
                 activation="tanh",
                 debug=False)

    predictions = layer(inputs)

    loss = tf.losses.mean_squared_error(labels=outputs,
                                        predictions=predictions)

    loss_summary = tf.summary.scalar("loss", loss)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    opt = optimizer.minimize(loss)

    writer.add_graph(tf.get_default_graph())
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_value = None
    for i in range(epochs):
        _, loss_value = sess.run((opt, loss))
        if i % 10 == 0:
            summary = sess.run(loss_summary)
            writer.add_summary(summary, i)

        LOG.debug("epoch {}, loss: {}".format(i, loss_value))


    state = tf.constant(0, dtype=tf.float32)
    predictions = layer(inputs)

    return sess.run(predictions), float(loss_value)


if __name__ == "__main__":
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS

    activation = "tanh"

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width)
                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                # inputs, outputs_ = inputs[:3], outputs_[:3]
                # increase *units* in order to increase the capacity of the model
                for units in _units:
                    predictions, loss = fit(inputs, outputs_, units, activation, width, weight)
                    fname = constants.FNAME_FORMAT["plays_loss"].format(method=method, weight=weight,
                                                                        width=width, activation=activation, units=units)
                    tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                    fname = constants.FNAME_FORMAT["plays_predictions"].format(method=method, weight=weight,
                                                                               width=width, activation=activation, units=units)
                    tdata.DatasetSaver.save_data(inputs, predictions, fname)
