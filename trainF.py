import sys
import argparse
import time
import numpy as np
import tensorflow as tf

import utils
from core import Play, MyModel
import log as logging
import constants
import trading_data as tdata

constants.LOG_DIR = "./log/plays"
writer = utils.get_tf_summary_writer("./log/plays")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS
points = constants.POINTS


def fit(inputs, outputs, units, activation, width, true_weight, loss='mse', mu=0, sigma=0.01, loss_file_name="./tmp/trainF-loss.csv"):
    mu = float(mu)
    sigma = float(sigma)
    fname = constants.FNAME_FORMAT['mc'].format(mu=mu, sigma=sigma, points=inputs.shape[-1])
    try:
        B, _ = tdata.DatasetLoader.load_data(fname)
    except:
        B = tdata.DatasetGenerator.systhesis_markov_chain_generator(inputs.shape[-1], mu, sigma)
        fname = constants.FNAME_FORMAT['mc'].format(points=inputs.shape[-1], mu=mu, sigma=sigma)
        tdata.DatasetSaver.save_data(B, B, fname)

    units = units
    batch_size = 10
    epochs = EPOCHS // batch_size
    epochs = 1
    steps_per_epoch = batch_size

    train_inputs, train_outputs = inputs, outputs

    import time
    start = time.time()
    nb_plays = 1
    batch_size = 1
    play = MyModel(batch_size=batch_size,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays)
    play.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    predictions = play.predict(inputs)

    prices = play.predict(B)
    B = B.reshape(-1)
    prices = prices.reshape(-1)
    return B, prices, predictions


if __name__ == "__main__":
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", dest="loss",
                        required=True)
    parser.add_argument("--mu", dest="mu",
                        required=False,
                        type=float)
    parser.add_argument("--sigma", dest="sigma",
                        required=False,
                        type=float)
    parser.add_argument("--units", dest="units",
                        required=False,
                        type=int)

    argv = parser.parse_args(sys.argv[1:])

    loss_name = argv.loss
    mu = argv.mu or 0
    sigma = argv.sigma or 0.01
    units = argv.units or 1

    activation = "tanh"

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points)
                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                inputs, outputs_ = outputs_, inputs  # F neural network
                inputs, outputs_ = inputs[:40], outputs_[:40]
                # increase *units* in order to increase the capacity of the model
                # for units in _units:
                if True:
                    loss_file_name = constants.FNAME_FORMAT['F_loss_history'].format(method=method,
                                                                                     weight=weight,
                                                                                     activation=activation,
                                                                                     units=units,
                                                                                     width=width,
                                                                                     mu=mu,
                                                                                     sigma=sigma,
                                                                                     points=points,
                                                                                     loss=loss_name)
                    B, prices, predictions = fit(inputs, outputs_, units, activation, width, weight, loss_name, mu=mu, sigma=sigma, loss_file_name=loss_file_name)

                    fname = constants.FNAME_FORMAT['F'].format(method=method,
                                                               weight=weight,
                                                               activation=activation,
                                                               units=units,
                                                               width=width,
                                                               mu=mu,
                                                               sigma=sigma,
                                                               points=points,
                                                               loss=loss_name)
                    tdata.DatasetSaver.save_data(inputs, predictions, fname)
                    fname = constants.FNAME_FORMAT['F_predictions'].format(method=method,
                                                                           weight=weight,
                                                                           activation=activation,
                                                                           units=units,
                                                                           width=width,
                                                                           mu=mu,
                                                                           sigma=sigma,
                                                                           points=points,
                                                                           loss=loss_name)

                    tdata.DatasetSaver.save_data(B, prices, fname)
