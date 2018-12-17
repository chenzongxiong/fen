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
batch_sizes = [1600]


def operator_generator():
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["operators"].format(method=method, weight=weight, width=width)
                inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                fname = constants.FNAME_FORMAT["operators_predictions"].format(method=method, weight=weight, width=width)
                _, predictions = tdata.DatasetLoader.load_data(fname)
                inputs = np.vstack([inputs, inputs]).T
                outputs = np.vstack([ground_truth, predictions]).T
                colors = utils.generate_colors(outputs.shape[-1])
                fname = constants.FNAME_FORMAT["operators_gif"].format(method=method, weight=weight, width=width)
                utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                fname = constants.FNAME_FORMAT["operators_gif_snake"].format(method=method, weight=weight, width=width)
                utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def play_generator():
    activation = "tanh"

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width)
                _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                inputs = np.vstack([_inputs, _inputs]).T
                for _units in units:
                    fname = constants.FNAME_FORMAT["plays_predictions"].format(method=method, weight=weight,
                                                                               width=width, activation=activation,
                                                                               units=_units)
                    _, predictions = tdata.DatasetLoader.load_data(fname)
                    outputs = np.vstack([ground_truth, predictions]).T
                    colors = utils.generate_colors(outputs.shape[-1])
                    fname = constants.FNAME_FORMAT["plays_gif"].format(method=method, weight=weight, width=width,
                                                                       activation=activation, units=_units)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                    fname = constants.FNAME_FORMAT["plays_gif_snake"].format(method=method, weight=weight, width=width,
                                                                             activation=activation, units=_units)
                    utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")


def model_generator():
    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                    fname = constants.FNAME_FORMAT["models"].format(method=method, weight=weight,
                                                                    width=width, nb_plays=_nb_plays)
                    _inputs, ground_truth = tdata.DatasetLoader.load_data(fname)
                    for __nb_plays in nb_plays:
                        for bz in batch_sizes:
                            fname = constants.FNAME_FORMAT["models_predictions"].format(method=method, weight=weight,
                                                                                        width=width, nb_plays=_nb_plays,
                                                                                        nb_plays_=__nb_plays,
                                                                                        batch_size=bz)
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
                                                                                batch_size=bz)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                            fname = constants.FNAME_FORMAT["models_gif_snake"].format(method=method, weight=weight,
                                                                                      width=width, nb_plays=_nb_plays,
                                                                                      nb_plays_=__nb_plays,
                                                                                      batch_size=bz)
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
                    for __nb_plays in nb_plays:
                        for bz in batch_sizes:
                            fname = constants.FNAME_FORMAT["models_F_predictions"].format(method=method, weight=weight,
                                                                                          width=width, nb_plays=_nb_plays,
                                                                                          nb_plays_=__nb_plays,
                                                                                          batch_size=bz)
                            try:
                                _, predictions = tdata.DatasetLoader.load_data(fname)
                            except:
                                continue

                            outputs = np.vstack([ground_truth, predictions]).T
                            colors = utils.generate_colors(outputs.shape[-1])
                            inputs = np.vstack([_inputs for _ in range(outputs.shape[-1])]).T
                            fname = constants.FNAME_FORMAT["models_F_gif"].format(method=method, weight=weight,
                                                                                  width=width, nb_plays=_nb_plays,
                                                                                  nb_plays_=__nb_plays,
                                                                                  batch_size=bz)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors)
                            fname = constants.FNAME_FORMAT["models_F_gif_snake"].format(method=method, weight=weight,
                                                                                        width=width, nb_plays=_nb_plays,
                                                                                        nb_plays_=__nb_plays,
                                                                                        batch_size=bz)
                            utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode="snake")




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
    argv = parser.parse_args(sys.argv[1:])

    if argv.operator:
        operator_generator()
    if argv.play:
        play_generator()
    if argv.model:
        model_generator()
    if argv.GF:
        GF_generator()
