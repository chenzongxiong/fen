import sys
import argparse
import numpy as np

import log as logging
import constants
import utils
import trading_data as tdata


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
    mu = 0
    # sigma = 0.01
    # sigma = 0.001
    # sigma = 2
    sigma = 2
    points = 1000
    # nb_plays = [20]
    nb_plays = 20
    nb_plays_ = 20

    step = 20

    units = 20
    period = 1
    interp = 1
    loss_name = 'mse'
    state = 0
    bz = 1

    for method in methods:
        for weight in weights:
            for width in widths:
                if True:
                    LOG.debug("Processing method: {}, weight: {}, width: {}, points: {}, units: {}, np_plays: {}, sigma: {}, mu: {}, loss: {}".format(method, weight, width, points, units, nb_plays, sigma, mu, loss_name))
                    fname = constants.FNAME_FORMAT["models_nb_plays_noise"].format(method=method,
                                                                                   weight=weight,
                                                                                   width=width,
                                                                                   nb_plays=nb_plays,
                                                                                   units=units,
                                                                                   points=points,
                                                                                   mu=mu,
                                                                                   sigma=sigma)
                    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                    diff = _inputs[1:] - _inputs[:-1]
                    _max = np.max(np.abs(diff))
                    # interp = (int)(_max / 0.2)
                    # if interp < 1:
                    #     interp = 1
                    # import ipdb; ipdb.set_trace()
                    if True:
                        if True:
                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_predictions"].format(method=method,
                                                                                                       weight=weight,
                                                                                                       width=width,
                                                                                                       nb_plays=nb_plays,
                                                                                                       nb_plays_=nb_plays_,
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

                            if interp != 1:
                                from scipy.interpolate import interp1d

                                t_ = np.linspace(1, points, points)

                                f1 = interp1d(t_, _inputs)
                                f2 = interp1d(t_, _inputs, kind='cubic')
                                t_interp = np.linspace(1, points, (int)(interp*points-interp+1))
                                # t_interp = np.linspace(1, points, (int)(interp*points))

                                _inputs_interp = np.interp(t_interp, t_, _inputs)
                                # _inputs_interp = f2(t_interp)

                                # ground_truth_interp = np.interp(_inputs_interp, _inputs, ground_truth, period=1)
                                # predictions_interp = np.interp(_inputs_interp, _inputs, predictions, period=1)
                                # ground_truth_interp =
                                # model = core.MyModel(nb_plays=nb_plays, units=units, debug=True, batch_size=batch_size, activation=None, timestep=points, input_dim=1)
                                _, ground_truth_interp = tdata.DatasetGenerator.systhesis_model_generator(nb_plays=nb_plays,
                                                                                                          units=units,
                                                                                                          inputs=_inputs_interp,
                                                                                                          points=t_interp.shape[0],
                                                                                                          mu=None,
                                                                                                          sigma=None)
                                predictions_interp = ground_truth_interp
                                # import matplotlib.pyplot as plt
                                # length = 50
                                # plt.plot(t_[:length], _inputs[:length], 'o')
                                # plt.plot(t_interp[:interp*length-1], _inputs_interp[:(interp*length-1)], '-x')
                                # plt.show()


                                # plt.plot(t_[:length], ground_truth[:length], 'o')
                                # plt.plot(t_interp[:interp*length-1], ground_truth_interp[:(interp*length-1)], '-x')
                                # plt.show()
                                # utils.save_animation(t_, _inputs, "./1.gif", step=points, colors=['black'])
                                # utils.save_animation(t_interp, _inputs_interp, "./1_interp.gif", step=points, colors=['black'])


                                _inputs = _inputs_interp
                                ground_truth = ground_truth_interp
                                predictions = predictions_interp


                                # utils.save_animation(t_, ground_truth, "./2.gif", step=points, colors=['black'])
                                # utils.save_animation(t_interp, ground_truth_interp, "./2_interp.gif", step=points, colors=['black'])

                                # _inputs = _inputs[:2000]
                                # ground_truth = ground_truth[:2000]
                                # predictions = predictions[:2000]

                                # fname = constants.FNAME_FORMAT['F_interp'].format(method=method,
                                #                                                   weight=weight,
                                #                                                   width=width,
                                #                                                   nb_plays=nb_plays,
                                #                                                   units=units,
                                #                                                   points=points,
                                #                                                   mu=mu,
                                #                                                   sigma=sigma,
                                #                                                   nb_plays_=nb_plays_,
                                #                                                   batch_size=1,
                                #                                                   state=state,
                                #                                                   loss=loss_name)

                                # tdata.DatasetSaver.save_data(ground_truth, _inputs, fname)
                                fname = constants.FNAME_FORMAT['models_nb_plays_noise_interp'].format(method=method,
                                                                                                      weight=weight,
                                                                                                      width=width,
                                                                                                      nb_plays=nb_plays,
                                                                                                      units=units,
                                                                                                      points=points,
                                                                                                      mu=mu,
                                                                                                      sigma=sigma,
                                                                                                      nb_plays_=nb_plays_,
                                                                                                      batch_size=1,
                                                                                                      state=state,
                                                                                                      loss=loss_name,
                                                                                                      interp=interp)
                                tdata.DatasetSaver.save_data(_inputs, ground_truth, fname)

                                import ipdb; ipdb.set_trace()

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_gif"].format(method=method,
                                                                                               weight=weight,
                                                                                               width=width,
                                                                                               nb_plays=nb_plays,
                                                                                               nb_plays_=nb_plays_,
                                                                                               batch_size=bz,
                                                                                               units=units,
                                                                                               points=points,
                                                                                               mu=mu,
                                                                                               sigma=sigma,
                                                                                               loss=loss_name)
                            # step = inputs.shape[0]
                            # utils.save_animation(inputs, outputs, fname, step=step, colors=colors)

                            _inputs = np.hstack([_inputs, _inputs])
                            ground_truth = np.hstack([ground_truth, ground_truth])
                            predictions = np.hstack([predictions, predictions])

                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_gif_snake"].format(method=method,
                                                                                                     weight=weight,
                                                                                                     width=width,
                                                                                                     nb_plays=nb_plays,
                                                                                                     nb_plays_=nb_plays_,
                                                                                                     batch_size=bz,
                                                                                                     units=units,
                                                                                                     points=points,
                                                                                                     mu=mu,
                                                                                                     sigma=sigma,
                                                                                                     loss=loss_name)
                            # step = 100
                            utils.save_animation(inputs, outputs, fname, step=step, colors=colors, mode="snake")

                            fname = constants.FNAME_FORMAT["models_nb_plays_noise_ts_outputs_gif"].format(method=method,
                                                                                                          weight=weight,
                                                                                                          width=width,
                                                                                                          nb_plays=nb_plays,
                                                                                                          nb_plays_=nb_plays_,
                                                                                                          batch_size=bz,
                                                                                                          units=units,
                                                                                                          points=points,
                                                                                                          mu=mu,
                                                                                                          sigma=sigma,
                                                                                                          loss=loss_name)

                            # _inputs = np.arange(points)
                            # inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            # utils.save_animation(inputs, outputs, fname, step=points, colors=colors)



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
