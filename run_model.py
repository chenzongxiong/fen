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


def fit(inputs,
        outputs,
        units=1,
        activation='tanh',
        nb_plays=1,
        batch_size=1,
        loss='mse',
        loss_file_name="./tmp/my_model_loss_history.csv",
        learning_rate=0.001, weights_name='model.h5'):

    epochs = 1
    steps_per_epoch = batch_size

    start = time.time()
    input_dim = 10
    timestep = inputs.shape[0] // input_dim
    agent = MyModel(input_dim=input_dim,
                    timestep=timestep,
                    units=units,
                    activation=activation,
                    nb_plays=nb_plays)
    # agent.load_weights(weights_fname)
    LOG.debug("Learning rate is {}".format(learning_rate))
    agent.fit(inputs, outputs, verbose=1, epochs=epochs,
              steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name, learning_rate=learning_rate)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    # agent.weights

    # agent.save_weights(weights_fname)
    # predictions = agent(inputs)

    predictions = agent.predict(inputs)
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss


def predict(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=1, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv", learning_rate=0.001, weights_name='model.h5'):

    steps_per_epoch = batch_size

    start = time.time()
    predictions_list = []
    input_dim = 100
    # timestep = 100 // input_dim
    timestep = inputs.shape[0] // input_dim
    timestep = 10
    start = time.time()
    agent = MyModel(batch_size=batch_size,
                    input_dim=input_dim,
                    timestep=timestep,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays)

    agent.load_weights(weights_fname)
    for i in range(9):
        LOG.debug("Predict on #{} sample".format(i+1))
        predictions = agent.predict(inputs[i*1000: (i+1)*1000])
        predictions_list.append(predictions)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")

    _predictions = np.hstack(predictions_list)
    loss = ((_predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    # import ipdb; ipdb.set_trace()

    return _predictions, loss


if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    # Hyper Parameters
    learning_rate = 0.1
    loss_name = 'mse'
    method = 'sin'
    with_noise = False

    mu = 0
    sigma = 2
    points = 1000
    input_dim = 1
    # ground truth
    nb_plays = 20
    units = 20
    state = 0
    activation = 'tanh'
    # predicitons
    __nb_plays__ = 20
    __units__ = 20
    __state__ = 0
    __activation__ = 'tanh'

    if with_noise is False:
        mu = 0
        sigma = 0

    LOG.debug("generate model data for method {}, units {}, nb_plays {}, mu: {}, sigma: {}, points: {}, activation: {}, input_dim: {}".format(method, units, nb_plays, mu, sigma, points, activation, input_dim))

    fname = constants.DATASET_PATH['models'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)
    inputs, grouth_truth = tdata.DatasetLoader.load_data(fname)


    loss_history_file = constants.DATASET_PATH["models_loss_history"].format(method=method,
                                                                             activation=activation,
                                                                            state=state,
                                                                             mu=mu,
                                                                             sigma=sigma,
                                                                             units=units,
                                                                             nb_plays=nb_plays,
                                                                             points=points,
                                                                             input_dim=input_dim,
                                                                            __activation__=__activation__,
                                                                             __state__=__state__,
                                                                            __units__=__units__,
                                                                            __nb_plays__=__nb_plays__,
                                                                             loss=loss_name)

    weights_fname = constants.DATASET_PATH["models_saved_weights"].format(method=method,
                                                                          activation=activation,
                                                                          state=state,
                                                                          mu=mu,
                                                                          sigma=sigma,
                                                                          units=units,
                                                                          nb_plays=nb_plays,
                                                                          points=points,
                                                                          input_dim=input_dim,
                                                                          __activation__=__activation__,
                                                                          __state__=__state__,
                                                                          __units__=__units__,
                                                                          __nb_plays__=__nb_plays__,
                                                                          loss=loss_name)

    predicted_fname = constants.DATASET_PATH["models_predictions"].format(method=method,
                                                                          activation=activation,
                                                                          state=state,
                                                                          mu=mu,
                                                                          sigma=sigma,
                                                                          units=units,
                                                                          nb_plays=nb_plays,
                                                                          points=points,
                                                                          input_dim=input_dim,
                                                                          __activation__=__activation__,
                                                                          __state__=__state__,
                                                                          __units__=__units__,
                                                                          __nb_plays__=__nb_plays__,
                                                                          loss=loss_name)

    predictions, loss = fit(inputs=inputs,
                            outputs=grouth_truth,
                            units=__units__,
                            activation=__activation__,
                            nb_plays=__nb_plays__,
                            loss=loss_name,
                            learning_rate=learning_rate,
                            loss_file_name=loss_history_file,
                            weights_name=weights_fname)

    tdata.DatasetSaver.save_data(inputs, predictions, predicted_fname)

    # for method in methods:
    #     for weight in weights:
    #         for width in widths:
    #             LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #             # fname = constants.FNAME_FORMAT["models_noise"].format(method=method,
    #             #                                                       weight=weight,
    #             #                                                       width=width,
    #             #                                                       nb_plays=nb_plays,
    #             #                                                       units=units,
    #             #                                                       mu=mu,
    #             #                                                       sigma=sigma,
    #             #                                                       points=points)
    #             # fname = constants.FNAME_FORMAT["models_nb_plays_noise"].format(method=method,
    #             #                                                                weight=weight,
    #             #                                                                width=width,
    #             #                                                                nb_plays=nb_plays,
    #             #                                                                units=units,
    #             #                                                                points=points,
    #             #                                                                mu=mu,
    #             #                                                                sigma=sigma)
    #             interp = 10
    #             fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp"].format(method=method,
    #                                                                                   weight=weight,
    #                                                                                   width=width,
    #                                                                                   nb_plays=nb_plays,
    #                                                                                   units=units,
    #                                                                                   points=points,
    #                                                                                   mu=mu,
    #                                                                                   sigma=sigma,
    #                                                                                   interp=interp)

    #             inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
    #             # import ipdb; ipdb.set_trace()
    #             inputs, outputs_ = inputs[:9000], outputs_[:9000]
    #             # increase *units* in order to increase the capacity of the model
    #             # for units in _units:
    #             if True:
    #                 # loss_history_file = constants.FNAME_FORMAT["models_noise_loss_history"].format(method=method,
    #                 #                                                                                weight=weight,
    #                 #                                                                                width=width,
    #                 #                                                                                nb_plays=nb_plays,
    #                 #                                                                                units=units,
    #                 #                                                                                mu=mu,
    #                 #                                                                                sigma=sigma,
    #                 #                                                                                nb_plays_=nb_plays_,
    #                 #                                                                                batch_size=batch_size,
    #                 #                                                                                loss=loss_name,
    #                 #                                                                                points=points)
    #                 # weights_fname = constants.FNAME_FORMAT["models_noise_saved_weights"].format(method=method,
    #                 #                                                                             weight=weight,
    #                 #                                                                             width=width,
    #                 #                                                                             nb_plays=nb_plays,
    #                 #                                                                             units=units,
    #                 #                                                                             mu=mu,
    #                 #                                                                             sigma=sigma,
    #                 #                                                                             nb_plays_=nb_plays_,
    #                 #                                                                             batch_size=batch_size,
    #                 #                                                                             loss=loss_name,
    #                 #                                                                             points=points)

    #                 loss_history_file = constants.FNAME_FORMAT["models_nb_plays_noise_loss_history"].format(method=method,
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
    #                 weights_fname = constants.FNAME_FORMAT["models_nb_plays_noise_saved_weights"].format(method=method,
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
    #                 # loss_history_file = constants.FNAME_FORMAT["models_nb_plays_noise_interp_loss_history"].format(method=method,
    #                 #                                                                                                weight=weight,
    #                 #                                                                                                width=width,
    #                 #                                                                                                nb_plays=nb_plays,
    #                 #                                                                                                units=units,
    #                 #                                                                                                mu=mu,
    #                 #                                                                                                sigma=sigma,
    #                 #                                                                                                nb_plays_=nb_plays_,
    #                 #                                                                                                batch_size=batch_size,
    #                 #                                                                                                loss=loss_name,
    #                 #                                                                                                points=points,
    #                 #                                                                                                interp=interp)
    #                 # weights_fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp_saved_weights"].format(method=method,
    #                 #                                                                                             interp=interp,
    #                 #                                                                                             weight=weight,
    #                 #                                                                                             width=width,
    #                 #                                                                                             nb_plays=nb_plays,
    #                 #                                                                                             units=units,
    #                 #                                                                                             mu=mu,
    #                 #                                                                                             sigma=sigma,
    #                 #                                                                                             nb_plays_=nb_plays_,
    #                 #                                                                                             batch_size=batch_size,
    #                 #                                                                                             loss=loss_name,
    #                 #                                                                                             points=points)



    #                 predictions, loss = fit(inputs, outputs_, units, activation, width, weight, method, nb_plays_, batch_size, loss_name, loss_history_file, learning_rate, weights_fname)
    #                 # fname = constants.FNAME_FORMAT["models_noise_loss"].format(method=method,
    #                 #                                                            weight=weight,
    #                 #                                                            width=width,
    #                 #                                                            nb_plays=nb_plays,
    #                 #                                                            units=units,
    #                 #                                                            mu=mu,
    #                 #                                                            sigma=sigma,
    #                 #                                                            nb_plays_=nb_plays_,
    #                 #                                                            batch_size=batch_size,
    #                 #                                                            loss=loss_name,
    #                 #                                                            points=points)

    #                 fname = constants.FNAME_FORMAT["models_nb_plays_noise_loss"].format(method=method,
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
    #                 # fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp_loss"].format(method=method,
    #                 #                                                                            interp=interp,
    #                 #                                                                            weight=weight,
    #                 #                                                                            width=width,
    #                 #                                                                            nb_plays=nb_plays,
    #                 #                                                                            units=units,
    #                 #                                                                            mu=mu,
    #                 #                                                                            sigma=sigma,
    #                 #                                                                            nb_plays_=nb_plays_,
    #                 #                                                                            batch_size=batch_size,
    #                 #                                                                            loss=loss_name,
    #                 #                                                                            points=points)
    #                 tdata.DatasetSaver.save_loss({"loss": loss}, fname)
    #                 # fname = constants.FNAME_FORMAT["models_noise_predictions"].format(method=method,
    #                 #                                                                   weight=weight,
    #                 #                                                                   width=width,
    #                 #                                                                   nb_plays=nb_plays,
    #                 #                                                                   units=units,
    #                 #                                                                   mu=mu,
    #                 #                                                                   sigma=sigma,
    #                 #                                                                   nb_plays_=nb_plays_,
    #                 #                                                                   batch_size=batch_size,
    #                 #                                                                   loss=loss_name,
    #                 #                                                                   points=points)

    #                 fname = constants.FNAME_FORMAT["models_nb_plays_noise_predictions"].format(method=method,
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

    #                 # fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp_predictions"].format(method=method,
    #                 #                                                                                   interp=interp,
    #                 #                                                                                   weight=weight,
    #                 #                                                                                   width=width,
    #                 #                                                                                   nb_plays=nb_plays,
    #                 #                                                                                   units=units,
    #                 #                                                                                   mu=mu,
    #                 #                                                                                   sigma=sigma,
    #                 #                                                                                   nb_plays_=nb_plays_,
    #                 #                                                                                   batch_size=batch_size,
    #                 #                                                                                   loss=loss_name,
    #                 #                                                                                   points=points)

    #                 tdata.DatasetSaver.save_data(inputs, predictions, fname)
