import sys
import argparse
import numpy as np
import tensorflow as tf

from core import Play
import utils
import trading_data as tdata
import log as logging
import constants


writer = utils.get_tf_summary_writer("./log/operators")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS

def fit(inputs, outputs, width, method, true_weight):
    LOG.debug("timestap is: {}".format(inputs.shape[0]))

    batch_size = 20
    epochs = 5000 // batch_size
    steps_per_epoch = batch_size
    units = 10

    play = Play(batch_size=batch_size,
                units=units,
                activation=None,
                network_type=constants.NetworkType.OPERATOR)

    play.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)

    LOG.debug("number of layer is: {}".format(play.number_of_layers))
    LOG.debug("weight: {}".format(play.weight))
    loss, metrics = play.evaluate(inputs, outputs, steps_per_epoch=steps_per_epoch)
    predictions = play.predict(inputs, steps_per_epoch=1)
    predictions = predictions.reshape(-1)
    return predictions, loss


if __name__ == '__main__':
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    # train dataset
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["operators"].format(method=method, weight=weight, width=width)
                inputs, outputs = tdata.DatasetLoader.load_data(fname)
                predictions, loss = fit(inputs, outputs, width, method, weight)

                fname = constants.FNAME_FORMAT["operators_loss"].format(method=method, weight=weight, width=width)
                tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                fname = constants.FNAME_FORMAT["operators_predictions"].format(method=method, weight=weight, width=width)
                tdata.DatasetSaver.save_data(inputs, predictions, fname)
