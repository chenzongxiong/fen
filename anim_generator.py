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
# batch_sizes = constants.BATCH_SIZE_LIST
batch_sizes = [100]
nb_plays = [20, 40]
points = constants.POINTS


def operator_generator():
    loss_name = 'mse'
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["operators"].format(method=method, weight=weight, width=width, points=points)
                inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                fname = constants.FNAME_FORMAT["operators_predictions"].format(method=method, weight=weight, width=width, points=points, loss=loss_name)
                _, predictions = tdata.DatasetLoader.load_data(fname)
                inputs = np.vstack([inputs, inputs]).T
                outputs = np.vstack([ground_truth, predictions]).T
                colors = utils.generate_colors(outputs.shape[-1])
                fname = constants.FNAME_FORMAT["operators_gif"].format(method=method, weight=weight, width=width, points=points, loss=loss_name)
                utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                fname = constants.FNAME_FORMAT["operators_gif_snake"].format(method=method, weight=weight, width=width, points=points, loss=loss_name)
                utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def play_generator():
    activation = "tanh"
    loss_name = 'mse'
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points)
                _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                inputs = np.vstack([_inputs, _inputs]).T
                for _units in units:
                    fname = constants.FNAME_FORMAT["plays_predictions"].format(method=method, weight=weight,
                                                                               width=width, activation=activation,
                                                                               units=_units, loss=loss_name,
                                                                               points=points)
                    _, predictions = tdata.DatasetLoader.load_data(fname)
                    outputs = np.vstack([ground_truth, predictions]).T
                    colors = utils.generate_colors(outputs.shape[-1])
                    fname = constants.FNAME_FORMAT["plays_gif"].format(method=method, weight=weight, width=width,
                                                                       activation=activation, units=_units,
                                                                       points=points, loss=loss_name)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                    fname = constants.FNAME_FORMAT["plays_gif_snake"].format(method=method, weight=weight, width=width,
                                                                             activation=activation, units=_units,
                                                                             loss=loss_name, points=points)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def model_generator():
    units = 8
    points = 500
    loss_name = 'mse'
    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                    fname = constants.FNAME_FORMAT["models"].format(method=method, weight=weight,
                                                                    width=width, nb_plays=_nb_plays, units=units, points=points)
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

                            # fname = constants.FNAME_FORMAT["models_predictions"].format(method=method, weight=weight,
                            #                                                             width=width, nb_plays=_nb_plays,
                            #                                                             nb_plays_=__nb_plays,
                            #                                                             batch_size=bz)
                            try:
                                _, predictions = tdata.DatasetLoader.load_data(fname)
                            except:
                                continue

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_gif"].format(method=method, weight=weight,
                                                                                width=width, nb_plays=_nb_plays,
                                                                                nb_plays_=__nb_plays,
                                                                                batch_size=bz, units=units, points=points, loss=loss_name)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                            fname = constants.FNAME_FORMAT["models_gif_snake"].format(method=method, weight=weight,
                                                                                      width=width, nb_plays=_nb_plays,
                                                                                      nb_plays_=__nb_plays,
                                                                                      batch_size=bz, units=units, points=points, loss=loss_name)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


                        # fname = constants.FNAME_FORMAT["models_multi"].format(method=method, weight=weight,
                        #                                                       width=width, nb_plays=_nb_plays)
                        # _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)

                        # for bz in batch_sizes:
                        #     fname = constants.FNAME_FORMAT["models_multi_predictions"].format(method=method, weight=weight,
                        #                                                                     width=width, nb_plays=_nb_plays,
                        #                                                                   batch_size=bz)
                        # _, predictions = tdata.DatasetLoader.load_data(fname)
                        # if _nb_plays == 1:
                        #     outputs = np.vstack([ground_truth, predictions]).T
                        # else:
                        #     outputs = np.hstack([ground_truth, predictions])

                        # colors = utils.generate_colors(outputs.shape[-1])
                        # inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                        # fname = constants.FNAME_FORMAT["models_multi_gif"].format(method=method, weight=weight,
                        #                                                           width=width, nb_plays=_nb_plays,
                        #                                                           batch_size=bz)
                        # utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                        # fname = constants.FNAME_FORMAT["models_multi_gif_snake"].format(method=method, weight=weight,
                        #                                                                 width=width, nb_plays=_nb_plays,
                        #                                                                 batch_size=bz)
                        # utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def GF_generator():
    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                    fname = constants.FNAME_FORMAT["models_F"].format(method=method, weight=weight,
                                                                      width=width, nb_plays=_nb_plays)
                    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                    import ipdb; ipdb.set_trace()
                    for __nb_plays in nb_plays:
                        for bz in batch_sizes:
                            fname = constants.FNAME_FORMAT["models_F_predictions"].format(method=method, weight=weight,
                                                                                          width=width, nb_plays=_nb_plays,
                                                                                          nb_plays_=__nb_plays,
                                                                                          batch_size=bz)
                            try:
                                _, predictions = tdata.DatasetLoader.load_data(fname)
                            except:
                                LOG.warn("fname {} not found.".format(fname))
                                continue

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_F_gif"].format(method=method, weight=weight,
                                                                                  width=width, nb_plays=_nb_plays,
                                                                                  nb_plays_=__nb_plays,
                                                                                  batch_size=bz)
                            utils.save_animation(inputs, outputs, fname, step=10, colors=colors)
                            fname = constants.FNAME_FORMAT["models_F_gif_snake"].format(method=method, weight=weight,
                                                                                        width=width, nb_plays=_nb_plays,
                                                                                        nb_plays_=__nb_plays,
                                                                                        batch_size=bz)
                            utils.save_animation(inputs, outputs, fname, step=10, colors=colors, mode="snake")


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
    _units = 20
    mu = 2.0
    sigma = 0.01

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points)
                # _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                _inputs1, ground_truth = tdata.DatasetLoader.load_data(fname)
                _inputs1, ground_truth = ground_truth[:1000], _inputs1[:1000]

                # inputs = np.vstack([_inputs, _inputs]).T
                # for _units in units:
                # _inputs1, ground_truth = _inputs1[:40], ground_truth[:40]
                if True:
                    fname = constants.FNAME_FORMAT["F"].format(method=method, weight=weight,
                                                               width=width, activation=activation,
                                                               units=_units, loss=loss_name,
                                                               mu=mu, sigma=sigma,
                                                               points=points)

                    _inputs2, predictions = tdata.DatasetLoader.load_data(fname)
                    inputs = np.vstack([_inputs1, _inputs2]).T
                    outputs = np.vstack([ground_truth, predictions]).T
                    colors = utils.generate_colors(outputs.shape[-1])
                    fname = constants.FNAME_FORMAT["F_gif"].format(method=method, weight=weight, width=width,
                                                                   activation=activation, units=_units,
                                                                   mu=mu, sigma=sigma,
                                                                   points=points, loss=loss_name)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                    fname = constants.FNAME_FORMAT["F_gif_snake"].format(method=method, weight=weight, width=width,
                                                                         activation=activation, units=_units,
                                                                         mu=mu, sigma=sigma,
                                                                         loss=loss_name, points=points)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def G_generator():
    activation = "tanh"
    loss_name = 'mse'
    # _units = 10
    units = 10
    mu = 0
    sigma = 0.01
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
    mu = 0
    sigma = 0.01
    points = 1000

    units = 20
    nb_plays = [20]
    loss_name = 'mse'
    state = 0

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
        model_generator_with_noise()
