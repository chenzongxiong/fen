import os
import sys
import argparse
import numpy as np

import log as logging
import constants
import utils
import trading_data as tdata
import colors as coloring

LOG = logging.getLogger(__name__)

methods = constants.METHODS
weights = constants.WEIGHTS
widths = constants.WIDTHS
units = constants.UNITS
nb_plays = constants.NB_PLAYS
batch_sizes = [100]
nb_plays = [20, 40]
points = constants.POINTS


def operator_generator_with_noise():
    mu = 0
    sigma = 0.1
    points = 5000
    loss_name = 'mse'
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}, points: {}".format(method, weight, width, points))
                fname = constants.FNAME_FORMAT["operators_noise"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points)
                inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                fname = constants.FNAME_FORMAT["operators_noise_predictions"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points, loss=loss_name)
                _, predictions = tdata.DatasetLoader.load_data(fname)
                inputs = np.vstack([inputs, inputs]).T
                outputs = np.vstack([ground_truth, predictions]).T
                colors = utils.generate_colors(outputs.shape[-1])
                fname = constants.FNAME_FORMAT["operators_noise_gif"].format(method=method, weight=weight, width=width, sigma=sigma, mu=mu, points=points, loss=loss_name)
                utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                fname = constants.FNAME_FORMAT["operators_noise_gif_snake"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points, loss=loss_name)
                utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def play_generator_with_noise():
    activation = "tanh"
    mu = 1
    sigma = 0.01
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays_noise"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma)
                _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                inputs = np.vstack([_inputs, _inputs]).T
                for _units in units:
                    fname = constants.FNAME_FORMAT["plays_noise_predictions"].format(method=method, weight=weight,
                                                                                     width=width, activation=activation,
                                                                                     units=_units, mu=mu, sigma=sigma)
                    _, predictions = tdata.DatasetLoader.load_data(fname)
                    outputs = np.vstack([ground_truth, predictions]).T
                    colors = utils.generate_colors(outputs.shape[-1])
                    fname = constants.FNAME_FORMAT["plays_noise_gif"].format(method=method, weight=weight, width=width,
                                                                             activation=activation, units=_units, mu=mu,
                                                                             sigma=sigma)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                    fname = constants.FNAME_FORMAT["plays_noise_gif_snake"].format(method=method, weight=weight, width=width,
                                                                                   activation=activation, units=_units,
                                                                                   mu=mu, sigma=sigma)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def F_generator():
    activation = "tanh"
    loss_name = 'mse'
    mu = 0
    sigma = 0.01
    sigma = 2
    nb_plays = 20
    units = 20
    points = 1000
    state = 0
    activation = 'tanh'
    nb_plays_ = 20
    step = 40
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}, mu: {}, sigma: {}, nb_plays: {}, points: {}, units: {}".format(method, weight, width, mu, sigma, nb_plays, points, units))
                # fname = constants.FNAME_FORMAT["models_noise"].format(method=method,
                #                                                       weight=weight,
                #                                                       width=width,
                #                                                       nb_plays=nb_plays,
                #                                                       units=units,
                #                                                       mu=mu,
                #                                                       sigma=sigma,
                #                                                       points=points)

                interp = 10
                fname = constants.FNAME_FORMAT["F_interp"].format(method=method,
                                                                  weight=weight,
                                                                  width=width,
                                                                  nb_plays=nb_plays,
                                                                  nb_plays_=nb_plays_,
                                                                  batch_size=1,
                                                                  units=units,
                                                                  mu=mu,
                                                                  sigma=sigma,
                                                                  points=points,
                                                                  state=state)
                # fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp"].format(method=method,
                #                                                                       weight=weight,
                #                                                                       width=width,
                #                                                                       nb_plays=nb_plays,
                #                                                                       units=units,
                #                                                                       points=points,
                #                                                                       mu=mu,
                #                                                                       sigma=sigma,
                #                                                                       interp=interp)

                _inputs1, ground_truth = tdata.DatasetLoader.load_data(fname)
                # _inputs1, ground_truth = ground_truth[:9000], _inputs1[:9000]
                _inputs1, ground_truth = _inputs1[:9000], ground_truth[:9000]
                # import ipdb; ipdb.set_trace()
                # inputs = np.vstack([_inputs, _inputs]).T
                # for _units in units:
                # _inputs1, ground_truth = _inputs1[:40], ground_truth[:40]
                if True:

                    # fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp_predictions"].format(method=method,
                    #                                                                                   interp=interp,
                    #                                                                                   weight=weight,
                    #                                                                                   width=width,
                    #                                                                                   nb_plays=nb_plays,
                    #                                                                                   units=units,
                    #                                                                                   mu=mu,
                    #                                                                                   sigma=sigma,
                    #                                                                                   nb_plays_=nb_plays_,
                    #                                                                                   batch_size=1,
                    #                                                                                   loss=loss_name,
                    #                                                                                   points=points)


                    fname = constants.FNAME_FORMAT["F_interp_predictions"].format(method=method,
                                                               weight=weight,
                                                               width=width,
                                                               nb_plays=nb_plays,
                                                               nb_plays_=nb_plays_,
                                                               units=units,
                                                               loss=loss_name,
                                                               mu=mu,
                                                               sigma=sigma,
                                                               batch_size=1,
                                                               state=state,
                                                               points=points)

                    _inputs2, predictions = tdata.DatasetLoader.load_data(fname)
                    inputs = np.vstack([_inputs1, _inputs2]).T
                    outputs = np.vstack([ground_truth, predictions]).T
                    colors = utils.generate_colors(outputs.shape[-1])
                    fname = constants.FNAME_FORMAT["F_interp_gif"].format(method=method,
                                                                   weight=weight,
                                                                   width=width,
                                                                   nb_plays=nb_plays,
                                                                   units=units,
                                                                   mu=mu,
                                                                   sigma=sigma,
                                                                   points=points,
                                                                   nb_plays_=nb_plays_,
                                                                   batch_size=1,
                                                                   state=state,
                                                                   loss=loss_name)
                    # fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp_gif"].format(method=method,
                    #                                                                           interp=interp,
                    #                                                weight=weight,
                    #                                                width=width,
                    #                                                nb_plays=nb_plays,
                    #                                                units=units,
                    #                                                mu=mu,
                    #                                                sigma=sigma,
                    #                                                points=points,
                    #                                                nb_plays_=nb_plays_,
                    #                                                batch_size=1,
                    #                                                state=state,
                    #                                                loss=loss_name)

                    step = 9000
                    # utils.save_animation(inputs, outputs, fname, step=step, colors=colors)
                    fname = constants.FNAME_FORMAT["F_interp_gif_snake"].format(method=method,
                                                                                interp=interp,
                                                                         weight=weight,
                                                                         width=width,
                                                                         nb_plays=nb_plays,
                                                                         nb_plays_=nb_plays_,
                                                                         units=units,
                                                                         mu=mu,
                                                                         sigma=sigma,
                                                                         loss=loss_name,
                                                                         state=state,
                                                                         batch_size=1,
                                                                         points=points)

                    # fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp_gif_snake"].format(method=method,
                    #                                                                                 interp=interp,
                    #                                                      weight=weight,
                    #                                                      width=width,
                    #                                                      nb_plays=nb_plays,
                    #                                                      nb_plays_=nb_plays_,
                    #                                                      units=units,
                    #                                                      mu=mu,
                    #                                                      sigma=sigma,
                    #                                                      loss=loss_name,
                    #                                                      state=state,
                    #                                                      batch_size=1,
                    #                                                      points=points)
                    # utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


                    fname = constants.FNAME_FORMAT["F_interp_ts_outputs_gif"].format(method=method,
                                                                                     weight=weight,
                                                                                     width=width,
                                                                                     nb_plays=nb_plays,
                                                                                     nb_plays_=nb_plays_,
                                                                                     batch_size=1,
                                                                                     units=units,
                                                                                     points=points,
                                                                                     mu=mu,
                                                                                     sigma=sigma,
                                                                                     state=state,
                                                                                     loss=loss_name)

                    _inputs = np.arange(9000)
                    inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                    utils.save_animation(inputs, outputs, fname, step=9000, colors=colors)


def G_generator():
    activation = "tanh"
    loss_name = 'mse'
    # _units = 10
    units = 20
    mu = 0
    sigma = 0.001
    nb_plays_ = 4
    batch_size = 10
    points = 500
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                # fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points
                # fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points)
                # inputs, outputs_ = tdata.DatasetLoader.load_data(fname)


                fname = constants.FNAME_FORMAT["F_predictions"].format(method=method, weight=weight, width=width, points=points, activation='tanh', units=units, sigma=sigma, mu=mu, loss='mse')
                # fname = constants.FNAME_FORMAT["F"].format(method=method, weight=weight, width=width, points=points, activation='tanh', units=units, sigma=sigma, mu=mu, loss='mse')
                print("F_predictions: ", fname)
                # _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                _inputs1, ground_truth = tdata.DatasetLoader.load_data(fname)
                _inputs1, ground_truth = ground_truth[:1000], _inputs1[:1000]
                # _inputs1, ground_truth = _inputs1[:1000], ground_truth[:1000]

                # inputs = np.vstack([_inputs, _inputs]).T
                # for _units in units:
                # _inputs1, ground_truth = _inputs1[:40], ground_truth[:40]
                if True:
                    # fname = constants.FNAME_FORMAT["F"].format(method=method, weight=weight,
                    #                                            width=width, activation=activation,
                    #                                            units=_units, loss=loss_name,
                    #                                            mu=mu, sigma=sigma,
                    #                                            points=points)
                    fname = constants.FNAME_FORMAT["G_predictions"].format(method=method, weight=weight,
                                                                           width=width, activation=activation, units=units,
                                                                           nb_plays=nb_plays, nb_plays_=nb_plays_, batch_size=batch_size,
                                                                           points=points, loss='mse', mu=mu, sigma=sigma)
                    print("G_predictions: ", fname)
                    _inputs2, predictions = tdata.DatasetLoader.load_data(fname)
                    inputs = np.vstack([_inputs1, _inputs2]).T
                    outputs = np.vstack([ground_truth, predictions]).T
                    colors = utils.generate_colors(outputs.shape[-1])
                    fname = constants.FNAME_FORMAT["G_gif"].format(method=method, weight=weight, width=width,
                                                                   activation=activation, units=units,
                                                                   mu=mu, sigma=sigma,
                                                                   points=points, loss=loss_name)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                    fname = constants.FNAME_FORMAT["G_gif_snake"].format(method=method, weight=weight, width=width,
                                                                         activation=activation, units=units,
                                                                         mu=mu, sigma=sigma,
                                                                         loss=loss_name, points=points)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def model_generator_with_noise():
    mu = 0
    sigma = 0.01
    points = 5000
    nb_plays = [40]

    units = 20

    loss_name = 'mse'

    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    LOG.debug("Processing method: {}, weight: {}, width: {}, points: {}, units: {}, np_plays: {}, sigma: {}, mu: {}, loss: {}".format(method, weight, width, points, units, nb_plays, sigma, mu, loss_name))
                    fname = constants.FNAME_FORMAT["models_noise"].format(method=method,
                                                                          weight=weight,
                                                                          width=width,
                                                                          nb_plays=_nb_plays,
                                                                          units=units,
                                                                          points=points,
                                                                          mu=mu,
                                                                          sigma=sigma)
                    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                    # for __nb_plays in nb_plays:
                    #     for bz in batch_sizes:
                    __nb_plays = _nb_plays
                    bz = 1
                    if True:
                        if True:
                            # fname = constants.FNAME_FORMAT["models_predictions"].format(method=method, weight=weight,
                            #                                                             width=width, nb_plays=_nb_plays,
                            #                                                             nb_plays_=__nb_plays,
                                                                                        # batch_size=bz)

                            fname = constants.FNAME_FORMAT["models_noise_predictions"].format(method=method,
                                                                                              weight=weight,
                                                                                              width=width,
                                                                                              nb_plays=_nb_plays,
                                                                                              nb_plays_=__nb_plays,
                                                                                              batch_size=bz,
                                                                                              units=units,
                                                                                              points=points,
                                                                                              mu=mu,
                                                                                              sigma=sigma,
                                                                                              loss=loss_name)
                            try:
                                _, predictions = tdata.DatasetLoader.load_data(fname)
                            except:
                                continue

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_noise_gif"].format(method=method,
                                                                                      weight=weight,
                                                                                      width=width,
                                                                                      nb_plays=_nb_plays,
                                                                                      nb_plays_=__nb_plays,
                                                                                      batch_size=bz,
                                                                                      units=units,
                                                                                      points=points,
                                                                                      mu=mu,
                                                                                      sigma=sigma,
                                                                                      loss=loss_name)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                            fname = constants.FNAME_FORMAT["models_noise_gif_snake"].format(method=method,
                                                                                            weight=weight,
                                                                                            width=width,
                                                                                            nb_plays=_nb_plays,
                                                                                            nb_plays_=__nb_plays,
                                                                                            batch_size=bz,
                                                                                            units=units,
                                                                                            points=points,
                                                                                            mu=mu,
                                                                                            sigma=sigma,
                                                                                            loss=loss_name)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")

                            fname = constants.FNAME_FORMAT["models_noise_ts_outputs_gif"].format(method=method,
                                                                                                 weight=weight,
                                                                                                 width=width,
                                                                                                 nb_plays=_nb_plays,
                                                                                                 nb_plays_=__nb_plays,
                                                                                                 batch_size=bz,
                                                                                                 units=units,
                                                                                                 points=points,
                                                                                                 mu=mu,
                                                                                                 sigma=sigma,
                                                                                                 loss=loss_name)
                            # steps = inputs.shape[-1]
                            _inputs = np.arange(points)
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            utils.save_animation(inputs, outputs, fname, step=points, colors=colors)



def model_noise_test_generator():
    sigma = 0.1
    points = 1000

    units = 20
    nb_plays = [20]

    state = 2

    mu = 0
    loss_name = 'mse'
    methods = ["mixed"]
    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    LOG.debug("Processing method: {}, weight: {}, width: {}, points: {}, units: {}, np_plays: {}, sigma: {}, mu: {}, loss: {}".format(method, weight, width, points, units, nb_plays, sigma, mu, loss_name))
                    fname = constants.FNAME_FORMAT["models_noise_test"].format(method=method, weight=weight,
                                                                               width=width, nb_plays=_nb_plays, units=units, points=points, mu=mu, sigma=sigma,
                                                                               state=state)
                    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                    # for __nb_plays in nb_plays:
                    #     for bz in batch_sizes:
                    __nb_plays = _nb_plays
                    bz = 1
                    if True:
                        if True:
                            fname = constants.FNAME_FORMAT["models_noise_test"].format(method=method, weight=weight,
                                                                                       width=width, nb_plays=_nb_plays, units=units, points=points, mu=mu, sigma=sigma,
                                                                                       state=state)
                            try:
                                fname = constants.FNAME_FORMAT["models_noise_test_predictions"].format(method=method,
                                                                                                       weight=weight,
                                                                                                       width=width,
                                                                                                       nb_plays=_nb_plays,
                                                                                                       nb_plays_=__nb_plays,
                                                                                                       batch_size=bz,
                                                                                                       units=units,
                                                                                                       points=points,
                                                                                                       mu=mu,
                                                                                                       sigma=sigma,
                                                                                                       loss=loss_name,
                                                                                                       state=state)

                                _, predictions = tdata.DatasetLoader.load_data(fname)
                            except:
                                continue

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])

                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_noise_test_gif"].format(method=method,
                                                                                           weight=weight,
                                                                                           width=width,
                                                                                           nb_plays=_nb_plays,
                                                                                           nb_plays_=__nb_plays,
                                                                                           batch_size=bz,
                                                                                           units=units,
                                                                                           points=points,
                                                                                           mu=mu,
                                                                                           sigma=sigma,
                                                                                           loss=loss_name,
                                                                                           state=state)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                            fname = constants.FNAME_FORMAT["models_noise_test_gif_snake"].format(method=method,
                                                                                                 weight=weight,
                                                                                                 width=width,
                                                                                                 nb_plays=_nb_plays,
                                                                                                 nb_plays_=__nb_plays,
                                                                                                 batch_size=bz,
                                                                                                units=units,
                                                                                                 points=points,
                                                                                                 mu=mu,
                                                                                                 sigma=sigma,
                                                                                                 loss=loss_name,
                                                                                                 state=state)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")

                            fname = constants.FNAME_FORMAT["models_noise_test_ts_outputs_gif"].format(method=method,
                                                                                                      weight=weight,
                                                                                                      width=width,
                                                                                                      nb_plays=_nb_plays,
                                                                                                      nb_plays_=__nb_plays,
                                                                                                      batch_size=bz,
                                                                                                      units=units,
                                                                                                      points=points,
                                                                                                      mu=mu,
                                                                                                      sigma=sigma,
                                                                                                      loss=loss_name,
                                                                                                      state=state)
                            _inputs = np.arange(points)
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            utils.save_animation(inputs, outputs, fname, step=points, colors=colors)



def model_nb_plays_generator_with_noise():
    step = 40
    method = 'sin'
    # method = 'noise'

    with_noise = True
    diff_weights = True
    run_test = False
    interp = 10
    force_rerun = False

    mu = 0
    sigma = 2
    points = 1000
    input_dim = 1
    # ground truth
    nb_plays = 20
    units = 20
    state = 0
    # activation = None
    activation = 'tanh'
    # predicitons
    __nb_plays__ = 20
    __units__ = 20
    __state__ = 0
    # __activation__ = None
    __activation__ = 'tanh'

    loss_name = 'mse'

    if method == 'noise':
        with_noise = True

    if with_noise is False:
        mu = 0
        sigma = 0

    if interp == 1:
        if run_test is False:
            if diff_weights is True:
                base_file_key = 'models_diff_weights'
                predictions_file_key = 'models_diff_weights_predictions'

                models_gif_key = 'models_diff_weights_gif'
                models_snake_gif_key = 'models_diff_weights_snake_gif'
                models_ts_outputs_gif_key = 'models_diff_weights_ts_outputs_gif'
            else:
                base_file_key = 'models'
                predictions_file_key = 'models_predictions'

                models_gif_key = 'models_gif'
                models_snake_gif_key = 'models_snake_gif'
                models_ts_outputs_gif_key = 'models_ts_outputs_gif'
        elif run_test is True:
            if diff_weights is True:
                base_file_key = 'models_diff_weights_test'
                predictions_file_key = 'models_diff_weights_test_predictions'
                models_gif_key = 'models_diff_weights_test_gif'
                models_snake_gif_key = 'models_diff_weights_test_snake_gif'
                models_ts_outputs_gif_key = 'models_diff_weights_test_ts_outputs_gif'
            else:
                raise
    elif interp != 1:
        if run_test is False:
            if diff_weights is True:
                base_file_key = 'models_diff_weights'
                models_interp_key = 'models_diff_weights_interp'
                predictions_file_key = 'models_diff_weights_predictions_interp'

                models_gif_key = 'models_diff_weights_interp_gif'
                models_snake_gif_key = 'models_diff_weights_snake_interp_gif'
                models_ts_outputs_gif_key = 'models_diff_weights_ts_outputs_interp_gif'
            else:
                # base_interp_key = 'models_interp'
                # predictions_file_key = 'models_predictions_interp'

                # models_gif_key = 'models_interp_gif'
                # models_snake_gif_key = 'models_snake_interp_gif'
                # models_ts_outputs_gif_key = 'models_ts_outputs_interp_gif'
                raise
        elif run_test is True:
            if diff_weights is True:
                base_file_key = 'models_diff_weights_test'
                models_interp_key = 'models_diff_weights_test_interp'
                predictions_file_key = 'models_diff_weights_test_predictions_interp'
                models_gif_key = 'models_diff_weights_test_interp_gif'
                models_snake_gif_key = 'models_diff_weights_test_snake_interp_gif'
                models_ts_outputs_gif_key = 'models_diff_weights_test_ts_outputs_interp_gif'
            else:
                raise

    if run_test is True and method == 'sin':
        method = 'mixed'
    # import ipdb; ipdb.set_trace()

    fname = constants.DATASET_PATH[base_file_key].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)

    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
    LOG.debug("Load **ground-truth** dataset from file: {}".format(coloring.cyan(fname)))

    predicted_fname = constants.DATASET_PATH[predictions_file_key].format(interp=interp,
                                                                          method=method,
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
    if interp == 1:
        try:
            _, predictions = tdata.DatasetLoader.load_data(predicted_fname)
            LOG.debug("Load **predicted** dataset from file: {}".format(coloring.cyan(predicted_fname)))
        except FileNotFoundError:
            LOG.warn("GROUND TRUTH and PREDICTIONS are the SAME dataset")
            predictions = ground_truth

    elif interp != 1:
        models_interp_fname = constants.DATASET_PATH[models_interp_key].format(interp=interp,
                                                                               method=method,
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

        if force_rerun is False and os.path.isfile(models_interp_fname):
            LOG.debug("Already interploted...")
            t_interp = np.linspace(1, points, (int)(interp*points-interp+1))
            _inputs_interp, ground_truth_interp = tdata.DatasetLoader.load_data(models_interp_fname)
            try:
                _, predictions_interp = tdata.DatasetLoader.load_data(predicted_fname)
                LOG.debug("Load **predicted** dataset from file: {}".format(coloring.cyan(predicted_fname)))
            except FileNotFoundError:
                LOG.warn("GROUND TRUTH and PREDICTIONS are the SAME dataset")
                predictions_interp = ground_truth_interp

            clip_length = min(predictions_interp.shape[0], _inputs_interp.shape[0])
            t_interp = t_interp[:clip_length]
            _inputs_interp = _inputs_interp[:clip_length]
            ground_truth_interp = ground_truth_interp[:clip_length]
            predictions_interp = predictions_interp[:clip_length]
        else:
            from scipy.interpolate import interp1d
            diff = _inputs[1:] - _inputs[:-1]
            LOG.debug("Max jump between two successive x is {}".format(np.max(np.abs(diff))))

            t_ = np.linspace(1, points, points)

            # f1 = interp1d(t_, _inputs)
            # f2 = interp1d(t_, _inputs, kind='cubic')
            t_interp = np.linspace(1, points, (int)(interp*points-interp+1))

            _inputs_interp = np.interp(t_interp, t_, _inputs)
            # _inputs_interp = f2(t_interp)
            clip_length = int((t_interp.shape[0] // input_dim) * input_dim)
            _inputs_interp = _inputs_interp[:clip_length]
            # ground_truth_interp = np.interp(_inputs_interp, _inputs, ground_truth, period=1)
            # predictions_interp = np.interp(_inputs_interp, _inputs, predictions, period=1)
            _, ground_truth_interp = tdata.DatasetGenerator.systhesis_model_generator(inputs=_inputs_interp,
                                                                                      nb_plays=nb_plays,
                                                                                      points=t_interp.shape[0],
                                                                                      units=units,
                                                                                      mu=None,
                                                                                      sigma=None,
                                                                                      input_dim=input_dim,
                                                                                      activation=activation,
                                                                                      with_noise=None,
                                                                                      method=None,
                                                                                      diff_weights=diff_weights)
            predictions_interp = ground_truth_interp
            # import matplotlib.pyplot as plt
            # length = 50
            # plt.plot(t_[:length], _inputs[:length], 'o')
            # plt.plot(t_interp[:interp*length-1], _inputs_interp[:(interp*length-1)], '-x')
            # plt.show()


            # plt.plot(t_[:length], ground_truth[:length], 'o')
            # plt.plot(t_interp[:interp*length-1], ground_truth_interp[:(interp*length-1)], '-x')
            # plt.show()

            LOG.debug("Save interploted dataset to file: {}".format(coloring.cyan(models_interp_fname)))
            tdata.DatasetSaver.save_data(_inputs_interp, ground_truth_interp, models_interp_fname)
            sys.exit(0)

        _inputs = _inputs_interp
        ground_truth = ground_truth_interp
        predictions = predictions_interp


    models_gif_fname = constants.DATASET_PATH[models_gif_key].format(interp=interp,
                                                                     method=method,
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
    models_snake_gif_fname = constants.DATASET_PATH[models_snake_gif_key].format(interp=interp,
                                                                                 method=method,
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
    models_ts_outputs_gif_fname = constants.DATASET_PATH[models_ts_outputs_gif_key].format(interp=interp,
                                                                                           method=method,
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

    LOG.debug("Write outputs vs. inputs {} into file {}".format(coloring.red("(sequence mode)"), coloring.cyan(models_gif_fname)))


    outputs = np.vstack([ground_truth, predictions]).T
    colors = utils.generate_colors(outputs.shape[-1])
    inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T

    # utils.save_animation(inputs, outputs, models_gif_fname, step=step, colors=colors)

    ##### SNAKE
    _inputs = np.hstack([_inputs, _inputs])
    ground_truth = np.hstack([ground_truth, ground_truth])
    predictions = np.hstack([predictions, predictions])

    inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
    outputs_snake = np.vstack([ground_truth, predictions]).T

    LOG.debug("Write outputs vs. inputs {} into file {}".format(coloring.red("(snake mode)"), coloring.cyan(models_snake_gif_fname)))
    utils.save_animation(inputs, outputs_snake, models_snake_gif_fname, step=step, colors=colors, mode="snake")


    if interp == 1:
        _inputs = np.arange(points)
    else:
        _inputs = t_interp

    inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
    # outputs = np.vstack([ground_truth, predictions]).T
    LOG.debug("Write outputs vs. ts into file {}".format(coloring.cyan(models_ts_outputs_gif_fname)))
    utils.save_animation(inputs, outputs, models_ts_outputs_gif_fname, step=points, colors=colors)



def model_nb_plays_noise_test_generator():
    sigma = 0.1
    points = 5000

    units = 20
    nb_plays = [20]

    state = 2

    mu = 0
    loss_name = 'mse'
    methods = ["mixed"]
    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    LOG.debug("Processing method: {}, weight: {}, width: {}, points: {}, units: {}, np_plays: {}, sigma: {}, mu: {}, loss: {}".format(method, weight, width, points, units, nb_plays, sigma, mu, loss_name))
                    fname = constants.FNAME_FORMAT["models_nb_plays_noise_test"].format(method=method, weight=weight,
                                                                                        width=width,
                                                                                        nb_plays=_nb_plays,
                                                                                        units=units,
                                                                                        points=points,
                                                                                        mu=mu,
                                                                                        sigma=sigma,
                                                                                        state=state)
                    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                    # for __nb_plays in nb_plays:
                    #     for bz in batch_sizes:
                    __nb_plays = _nb_plays
                    bz = 1
                    if True:
                        if True:
                            # fname = constants.FNAME_FORMAT["models_nb_plays_noise_test"].format(method=method,
                            #                                                                     weight=weight,
                            #                                                                     width=width,
                            #                                                                     nb_plays=_nb_plays,
                            #                                                                     units=units,
                            #                                                                     points=points,
                            #                                                                     mu=mu,
                            #                                                                     sigma=sigma,
                            #                                                                     state=state)
                            try:
                                # fname = constants.FNAME_FORMAT["models_nb_plays_noise_test_predictions"].format(method=method,
                                                                                                                # weight=weight,
                                                                                                                # width=width,
                                                                                                                # nb_plays=_nb_plays,
                                                                                                                # nb_plays_=__nb_plays,
                                                                                                                # batch_size=bz,
                                                                                                                # units=units,
                                                                                                                # points=points,
                                                                                                                # mu=mu,
                                                                                                                # sigma=sigma,
                                                                                                                # loss=loss_name,
                                                                                                                # state=state)

                                _, predictions = tdata.DatasetLoader.load_data(fname)
                            except:
                                continue

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])

                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_test_gif"].format(method=method,
                                                                                                    weight=weight,
                                                                                                    width=width,
                                                                                                    nb_plays=_nb_plays,
                                                                                                    nb_plays_=__nb_plays,
                                                                                                    batch_size=bz,
                                                                                                    units=units,
                                                                                                    points=points,
                                                                                                    mu=mu,
                                                                                                    sigma=sigma,
                                                                                                    loss=loss_name,
                                                                                                    state=state)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_test_gif_snake"].format(method=method,
                                                                                                          weight=weight,
                                                                                                          width=width,
                                                                                                          nb_plays=_nb_plays,
                                                                                                          nb_plays_=__nb_plays,
                                                                                                          batch_size=bz,
                                                                                                          units=units,
                                                                                                          points=points,
                                                                                                          mu=mu,
                                                                                                          sigma=sigma,
                                                                                                          loss=loss_name,
                                                                                                          state=state)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")

                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_test_ts_outputs_gif"].format(method=method,
                                                                                                                weight=weight,
                                                                                                                width=width,
                                                                                                                nb_plays=_nb_plays,
                                                                                                                nb_plays_=__nb_plays,
                                                                                                                batch_size=bz,
                                                                                                                units=units,
                                                                                                                points=points,
                                                                                                                mu=mu,
                                                                                                                sigma=sigma,
                                                                                                                loss=loss_name,
                                                                                                                state=state)
                            _inputs = np.arange(points)
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            utils.save_animation(inputs, outputs, fname, step=points, colors=colors)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", dest="operator",
                        required=False,
                        action="store_true",
                        help="generate operators' dataset")
    parser.add_argument("--play", dest="play",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")
    parser.add_argument("--model", dest="model",
                        required=False,
                        action="store_true",
                        help="generate models' dataset")
    parser.add_argument("--GF", dest="GF",
                        required=False,
                        action="store_true",
                        help="generate G & F's dataset")
    parser.add_argument("--operator-noise", dest="operator_noise",
                        required=False,
                        action="store_true")
    parser.add_argument("--play-noise", dest="play_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")
    parser.add_argument("--model-noise", dest="model_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")

    parser.add_argument("-F", dest="F",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")

    parser.add_argument("-G", dest="G",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")


    argv = parser.parse_args(sys.argv[1:])

    if argv.operator:
        operator_generator()
    if argv.play:
        play_generator()
    if argv.model:
        model_generator()
    if argv.GF:
        GF_generator()
    if argv.operator_noise:
       operator_generator_with_noise()
    if argv.play_noise:
        play_generator_with_noise()
    if argv.F:
        F_generator()
    if argv.G:
        G_generator()
    if argv.model_noise:
        # model_generator_with_noise()
        # model_noise_test_generator()
        model_nb_plays_generator_with_noise()
        # model_nb_plays_noise_test_generator()
