import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from core import MyModel
import utils
import trading_data as tdata
import log as logging
import constants
import colors

sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS


def fit(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=1, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv", learning_rate=0.001, weights_name='model.h5'):

    # epochs = EPOCHS // batch_size
    epochs = 250
    steps_per_epoch = batch_size

    start = time.time()
    agent = MyModel(batch_size=batch_size,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays)
    agent.load_weights(weights_fname)
    LOG.debug("Learning rate is {}".format(learning_rate))
    agent.fit(inputs, outputs, verbose=1, epochs=epochs,
              steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name, learning_rate=learning_rate)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    agent.weights

    agent.save_weights(weights_fname)
    # predictions = agent(inputs)

    predictions = agent.predict(inputs)
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss


def predict(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=1, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv", learning_rate=0.001, weights_name='model.h5'):
    start = time.time()
    agent = MyModel(batch_size=batch_size,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays)

    agent.load_weights(weights_fname)
    predictions = agent.predict(inputs)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss

if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", dest="loss",
                        required=False)
    parser.add_argument("--units", dest="units",
                        required=False, type=int)

    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS
    # batch_size = constants.BATCH_SIZE_LIST[0]
    batch_size = 1
    points = constants.POINTS
    # not test (40, 100), (40, 1000), (40, 5000)
    loss_name = 'mse'

    points = 1000
    nb_plays = 20
    nb_plays_ = nb_plays
    # nb_plays_ = 1
    # train dataset
    mu = 0
    sigma = 0.1
    state = 2

    activation = 'tanh'

    # units = argv.units
    units = 20

    # learning_rate = 0.01
    learning_rate = 0.1

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = constants.FNAME_FORMAT["models_noise"].format(method=method,
    #                                                                   weight=weight,
    #                                                                   width=width,
    #                                                                   nb_plays=nb_plays,
    #                                                                   units=units,
    #                                                                   mu=mu,
    #                                                                   sigma=sigma,
    #                                                                   points=points)

    #             inputs, outputs_ = tdata.DatasetLoader.load_data(fname)

    #             # inputs, outputs_ = inputs[:40], outputs_[:40]
    #             # increase *units* in order to increase the capacity of the model
    #             # for units in _units:
    #             if True:
    #                 loss_history_file = constants.FNAME_FORMAT["models_noise_loss_history"].format(method=method,
    #                                                                                                weight=weight,
    #                                                                                                width=width,
    #                                                                                                nb_plays=nb_plays,
    #                                                                                                units=units,
    #                                                                                                mu=mu,
    #                                                                                                sigma=sigma,
    #                                                                                                nb_plays_=nb_plays_,
    #                                                                                                batch_size=batch_size,
    #                                                                                                loss=loss_name,
    #                                                                                                points=points)
    #                 weights_fname = constants.FNAME_FORMAT["models_noise_saved_weights"].format(method=method,
    #                                                                                             weight=weight,
    #                                                                                             width=width,
    #                                                                                             nb_plays=nb_plays,
    #                                                                                             units=units,
    #                                                                                             mu=mu,
    #                                                                                             sigma=sigma,
    #                                                                                             nb_plays_=nb_plays_,
    #                                                                                             batch_size=batch_size,
    #                                                                                             loss=loss_name,
    #                                                                                             points=points)

    #                 predictions, loss = fit(inputs, outputs_, units, activation, width, weight, method, nb_plays_, batch_size, loss_name, loss_history_file, learning_rate, weights_fname)
    #                 fname = constants.FNAME_FORMAT["models_noise_loss"].format(method=method,
    #                                                                            weight=weight,
    #                                                                            width=width,
    #                                                                            nb_plays=nb_plays,
    #                                                                            units=units,
    #                                                                            mu=mu,
    #                                                                            sigma=sigma,
    #                                                                            nb_plays_=nb_plays_,
    #                                                                            batch_size=batch_size,
    #                                                                            loss=loss_name,
    #                                                                            points=points)
    #                 tdata.DatasetSaver.save_loss({"loss": loss}, fname)
    #                 fname = constants.FNAME_FORMAT["models_noise_predictions"].format(method=method,
    #                                                                                   weight=weight,
    #                                                                                   width=width,
    #                                                                                   nb_plays=nb_plays,
    #                                                                                   units=units,
    #                                                                                   mu=mu,
    #                                                                                   sigma=sigma,
    #                                                                                   nb_plays_=nb_plays_,
    #                                                                                   batch_size=batch_size,
    #                                                                                   loss=loss_name,
    #                                                                                   points=points)
    #                 tdata.DatasetSaver.save_data(inputs, predictions, fname)



    ################################################################################
    # methods = ["mixed"]

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             fname = constants.FNAME_FORMAT["models_noise_test"].format(method=method,
    #                                                                        weight=weight,
    #                                                                        width=width,
    #                                                                        nb_plays=nb_plays,
    #                                                                        units=units,
    #                                                                        mu=mu,
    #                                                                        sigma=sigma,
    #                                                                        points=points,
    #                                                                        state=state)

    #             weights_fname = constants.FNAME_FORMAT["models_noise_saved_weights"].format(method="sin",
    #                                                                                         weight=weight,
    #                                                                                         width=width,
    #                                                                                         nb_plays=nb_plays,
    #                                                                                         units=units,
    #                                                                                         mu=mu,
    #                                                                                         sigma=sigma,
    #                                                                                         nb_plays_=nb_plays_,
    #                                                                                         batch_size=batch_size,
    #                                                                                         loss=loss_name,
    #                                                                                         points=points)

    #             loss_history_file = "whatever"
    #             inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
    #             predictions, loss = predict(inputs, outputs_, units, activation, width, weight, method, nb_plays_, batch_size, loss_name, loss_history_file, learning_rate, weights_fname)

    #             fname = constants.FNAME_FORMAT["models_noise_test_loss"].format(method=method,
    #                                                                             weight=weight,
    #                                                                             width=width,
    #                                                                             nb_plays=nb_plays,
    #                                                                             units=units,
    #                                                                             mu=mu,
    #                                                                             sigma=sigma,
    #                                                                             nb_plays_=nb_plays_,
    #                                                                             batch_size=batch_size,
    #                                                                             loss=loss_name,
    #                                                                             points=points,
    #                                                                             state=state)
    #             tdata.DatasetSaver.save_loss({"loss": loss}, fname)
    #             fname = constants.FNAME_FORMAT["models_noise_test_predictions"].format(method=method,
    #                                                                                    weight=weight,
    #                                                                                    width=width,
    #                                                                                    nb_plays=nb_plays,
    #                                                                                    units=units,
    #                                                                                    mu=mu,
    #                                                                                    sigma=sigma,
    #                                                                                    nb_plays_=nb_plays_,
    #                                                                                    batch_size=batch_size,
    #                                                                                    loss=loss_name,
    #                                                                                    points=points,
    #                                                                                    state=state)
    #             tdata.DatasetSaver.save_data(inputs, predictions, fname)

    # ################################################################################
    # ## Different Weights

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             fname = constants.FNAME_FORMAT["models_nb_plays_noise"].format(method=method,
    #                                                                            weight=weight,
    #                                                                            width=width,
    #                                                                            nb_plays=nb_plays,
    #                                                                            units=units,
    #                                                                            mu=mu,
    #                                                                            sigma=sigma,
    #                                                                            points=points)

    #             inputs, outputs_ = tdata.DatasetLoader.load_data(fname)

    #             # inputs, outputs_ = inputs[:40], outputs_[:40]
    #             # increase *units* in order to increase the capacity of the model
    #             # for units in _units:
    #             if True:
    #                 loss_history_file = constants.FNAME_FORMAT["models_nb_plays_noise_loss_history"].format(method=method,
    #                                                                                                         weight=weight,
    #                                                                                                         width=width,
    #                                                                                                         nb_plays=nb_plays,
    #                                                                                                         units=units,
    #                                                                                                         mu=mu,
    #                                                                                                         sigma=sigma,
    #                                                                                                         nb_plays_=nb_plays_,
    #                                                                                                         batch_size=batch_size,
    #                                                                                                         loss=loss_name,
    #                                                                                                         points=points)
    #                 weights_fname = constants.FNAME_FORMAT["models_nb_plays_noise_saved_weights"].format(method=method,
    #                                                                                                      weight=weight,
    #                                                                                                      width=width,
    #                                                                                                      nb_plays=nb_plays,
    #                                                                                                      units=units,
    #                                                                                                      mu=mu,
    #                                                                                                      sigma=sigma,
    #                                                                                                      nb_plays_=nb_plays_,
    #                                                                                                      batch_size=batch_size,
    #                                                                                                      loss=loss_name,
    #                                                                                                      points=points)

    #                 predictions, loss = fit(inputs, outputs_, units, activation, width, weight, method, nb_plays_, batch_size, loss_name, loss_history_file, learning_rate, weights_fname)
    #                 fname = constants.FNAME_FORMAT["models_nb_plays_noise_loss"].format(method=method,
    #                                                                                     weight=weight,
    #                                                                                     width=width,
    #                                                                                     nb_plays=nb_plays,
    #                                                                                     units=units,
    #                                                                                     mu=mu,
    #                                                                                     sigma=sigma,
    #                                                                                     nb_plays_=nb_plays_,
    #                                                                                     batch_size=batch_size,
    #                                                                                     loss=loss_name,
    #                                                                                     points=points)
    #                 tdata.DatasetSaver.save_loss({"loss": loss}, fname)
    #                 fname = constants.FNAME_FORMAT["models_nb_plays_noise_predictions"].format(method=method,
    #                                                                                            weight=weight,
    #                                                                                            width=width,
    #                                                                                            nb_plays=nb_plays,
    #                                                                                            units=units,
    #                                                                                            mu=mu,
    #                                                                                            sigma=sigma,
    #                                                                                            nb_plays_=nb_plays_,
    #                                                                                            batch_size=batch_size,
    #                                                                                            loss=loss_name,
    #                                                                                            points=points)
    #                 tdata.DatasetSaver.save_data(inputs, predictions, fname)

    methods = ["mixed"]

    for method in methods:
        for weight in weights:
            for width in widths:
                fname = constants.FNAME_FORMAT["models_nb_plays_noise_test"].format(method=method,
                                                                                    weight=weight,
                                                                                    width=width,
                                                                                    nb_plays=nb_plays,
                                                                                    units=units,
                                                                                    mu=mu,
                                                                                    sigma=sigma,
                                                                                    points=points,
                                                                                    state=state)

                weights_fname = constants.FNAME_FORMAT["models_nb_plays_noise_saved_weights"].format(method="sin",
                                                                                                     weight=weight,
                                                                                                     width=width,
                                                                                                     nb_plays=nb_plays,
                                                                                                     units=units,
                                                                                                     mu=mu,
                                                                                                     sigma=sigma,
                                                                                                     nb_plays_=nb_plays_,
                                                                                                     batch_size=batch_size,
                                                                                                     loss=loss_name,
                                                                                                     points=points)

                loss_history_file = "whatever"
                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                predictions, loss = predict(inputs, outputs_, units, activation, width, weight, method, nb_plays_, batch_size, loss_name, loss_history_file, learning_rate, weights_fname)

                fname = constants.FNAME_FORMAT["models_nb_plays_noise_test_loss"].format(method=method,
                                                                                         weight=weight,
                                                                                         width=width,
                                                                                         nb_plays=nb_plays,
                                                                                         units=units,
                                                                                         mu=mu,
                                                                                         sigma=sigma,
                                                                                         nb_plays_=nb_plays_,
                                                                                         batch_size=batch_size,
                                                                                         loss=loss_name,
                                                                                         points=points,
                                                                                         state=state)
                tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                fname = constants.FNAME_FORMAT["models_nb_plays_noise_test_predictions"].format(method=method,
                                                                                                weight=weight,
                                                                                                width=width,
                                                                                                nb_plays=nb_plays,
                                                                                                units=units,
                                                                                                mu=mu,
                                                                                                sigma=sigma,
                                                                                                nb_plays_=nb_plays_,
                                                                                                batch_size=batch_size,
                                                                                                loss=loss_name,
                                                                                                points=points,
                                                                                                state=state)
                tdata.DatasetSaver.save_data(inputs, predictions, fname)
