# import sys
# import argparse
import numpy as np
import tensorflow as tf

import utils
from core import Play
import log as logging
import constants
import trading_data as tdata

writer = utils.get_tf_summary_writer("./log/plays")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS


def fit(inputs, outputs, units, activation, width, true_weight):

    units = units
    batch_size = 10
    epochs = 15000 // batch_size
    steps_per_epoch = batch_size

    import time
    start = time.time()
    play = Play(batch_size=batch_size,
                units=units,
                activation="tanh",
                network_type=constants.NetworkType.PLAY,
                loss='mse',
                debug=True)

    play.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("number of layer is: {}".format(play.number_of_layers))
    LOG.debug("weight: {}".format(play.weight))

    loss, metrics = play.evaluate(inputs, outputs, steps_per_epoch=steps_per_epoch)
    predictions = play.predict(inputs, steps_per_epoch=1)
    predicitons = predictions.reshape(-1)
    return predictions, loss


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
                # increase *units* in order to increase the capacity of the model
                for units in _units:
                    predictions, loss = fit(inputs, outputs_, units, activation, width, weight)
                    fname = constants.FNAME_FORMAT["plays_loss"].format(method=method, weight=weight,
                                                                        width=width, activation=activation, units=units)
                    tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                    fname = constants.FNAME_FORMAT["plays_predictions"].format(method=method, weight=weight,
                                                                               width=width, activation=activation, units=units)
                    tdata.DatasetSaver.save_data(inputs, predictions, fname)
