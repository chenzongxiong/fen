import sys
import argparse
import time
import numpy as np
import tensorflow as tf

import utils
from core import Play
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


def fit(inputs, outputs, units, activation, width, true_weight, loss='mse', mu=0, sigma=0.01):
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
    steps_per_epoch = batch_size

    train_inputs, train_outputs = inputs, outputs

    import time
    start = time.time()
    play = Play(batch_size=batch_size,
                units=units,
                activation="tanh",
                network_type=constants.NetworkType.PLAY,
                loss=loss,
                debug=False)

    # train F neural network
    if loss == 'mse':
        play.fit(train_inputs, train_outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)

        train_loss, metrics = play.evaluate(train_inputs, train_outputs, steps_per_epoch=steps_per_epoch)
        train_predictions = play.predict(train_inputs, steps_per_epoch=1)

        train_mu = train_sigma = test_mu = test_sigma = -1
    elif loss == 'mle':
        # play.fit2(train_inputs, mu, sigma, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
        # train_loss = test_loss = -1
        # train_predictions, train_mu, train_sigma = play.predict2(train_inputs, steps_per_epoch=1)
        # test_predictions, test_mu, test_sigma = play.predict2(test_inputs, steps_per_epoch=1)
        raise

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("number of layer is: {}".format(play.number_of_layers))
    LOG.debug("weight: {}".format(play.weight))

    train_predictions = train_predictions.reshape(-1)
    prices = play.predict(B)
    B = B.reshape(-1)
    prices = prices.reshape(-1)
    return B, prices, train_predictions


if __name__ == "__main__":
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", dest="loss",
                        required=True)
    parser.add_argument("--mu", dest="mu",
                        required=False)
    parser.add_argument("--sigma", dest="sigma",
                        required=False)

    argv = parser.parse_args(sys.argv[1:])

    loss_name = argv.loss
    mu = argv.mu or 0
    sigma = argv.sigma or 0.01

    activation = "tanh"

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points)
                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                inputs, outputs_ = outputs_, inputs  # F neural network
                # inputs, outputs_ = inputs[:40], outputs_[:40]
                # increase *units* in order to increase the capacity of the model
                for units in _units:
                    B, prices, predictions = fit(inputs, outputs_, units, activation, width, weight, loss_name, mu=mu, sigma=sigma)
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
