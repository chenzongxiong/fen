import sys
import argparse

import numpy as np
import trading_data as tdata
import log as logging
import constants


LOG = logging.getLogger(__name__)

methods = constants.METHODS
weights = constants.WEIGHTS
widths = constants.WIDTHS
NB_PLAYS = constants.NB_PLAYS
points = constants.POINTS

def operator_generator():
    states = [1, 4, 10 -1, -4, -10]

    for method in methods:
        for weight in weights:
            for width in widths:
                inputs = []
                outputs = []
                for state in states:
                    LOG.debug("Processing method: {}, weight: {}, width: {}, state: {}".format(method, weight, width, state))
                    inputs_, outputs_ = tdata.DatasetGenerator.systhesis_operator_generator(points=points,
                                                                                            weight=weight,
                                                                                            width=width,
                                                                                            state=state)
                    inputs.append(inputs_)
                    outputs.append(outputs_)
                fname = constants.FNAME_FORMAT['operators'].format(method=method, weight=weight, width=width)
                inputs = np.hstack(inputs)
                outputs = np.hstack(outputs)
                outputs = outputs.T
                tdata.DatasetSaver.save_data(inputs, outputs, fname)


def play_generator():
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                try:
                    inputs, _ = tdata.DatasetLoader.load_data(fname)
                except FileNotFoundError:
                    inputs = None

                inputs, outputs = tdata.DatasetGenerator.systhesis_play_generator(points=points, inputs=inputs)
                fname = constants.FNAME_FORMAT['plays'].format(method=method, weight=weight, width=width)
                tdata.DatasetSaver.save_data(inputs, outputs, fname)


def model_generator():
    for method in methods:
        for weight in weights:
            for width in widths:
                fname = "./training-data/plays/{}-{}-{}-tanh.csv".format(method, weight, width)
                try:
                    inputs, _ = tdata.DatasetLoader.load_data(fname)
                except FileNotFoundError:
                    inputs = None

                for nb_plays in NB_PLAYS:
                    inputs, outputs, plays_outputs = tdata.DatasetGenerator.systhesis_model_generator(
                        nb_plays=nb_plays, points=points, debug_plays=True, inputs=inputs)

                    fname = "./training-data/models/{}-{}-{}-{}.csv".format(method, weight, width, nb_plays)
                    tdata.DatasetSaver.save_data(inputs, outputs, fname)
                    fname = "./training-data/models/{}-{}-{}-{}-multi.csv".format(method, weight, width, nb_plays)
                    tdata.DatasetSaver.save_data(inputs, plays_outputs, fname)


def GF_generator():
    for method in methods:
        for weight in weights:
            for width in widths:
                for nb_plays in NB_PLAYS:
                    fname = constants.FNAME_FORMAT["models"].format(method=method, weight=weight, width=width, nb_plays=nb_plays)

                    try:
                        inputs, outputs = tdata.DatasetLoader.load_data(fname)
                        fname_G = constants.FNAME_FORMAT["models_G"].format(method=method, weight=weight, width=width, nb_plays=nb_plays)
                        fname_F = constants.FNAME_FORMAT["models_F"].format(method=method, weight=weight, width=width, nb_plays=nb_plays)
                        tdata.DatasetSaver.save_data(inputs, outputs, fname_G)
                        tdata.DatasetSaver.save_data(outputs, inputs, fname_F)
                    except FileNotFoundError:
                        LOG.warn("fname {} not found.".format(fname))


def operator_generator_with_noise():
    states = [1, 4, 10 -1, -4, -10]
    mu = 1
    sigma = 0.01
    for method in methods:
        for weight in weights:
            for width in widths:
                inputs = []
                outputs = []
                for state in states:
                    LOG.debug("Processing method: {}, weight: {}, width: {}, state: {}".format(method, weight, width, state))
                    inputs_, outputs_ = tdata.DatasetGenerator.systhesis_operator_generator(points=points,
                                                                                            weight=weight,
                                                                                            width=width,
                                                                                            state=state,
                                                                                            with_noise=True,
                                                                                            mu=mu,
                                                                                            sigma=sigma)
                    inputs.append(inputs_)
                    outputs.append(outputs_)
                fname = constants.FNAME_FORMAT['operators_noise'].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma)
                inputs = np.hstack(inputs)
                outputs = np.hstack(outputs)
                outputs = outputs.T
                tdata.DatasetSaver.save_data(inputs, outputs, fname)


def play_generator_with_noise():
    mu = 1
    sigma = 0.01
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT['operators_noise'].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma)
                try:
                    inputs, _ = tdata.DatasetLoader.load_data(fname)
                except FileNotFoundError:
                    inputs = None

                inputs, outputs = tdata.DatasetGenerator.systhesis_play_generator(points=points, inputs=inputs)
                fname = constants.FNAME_FORMAT['plays_noise'].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma)
                tdata.DatasetSaver.save_data(inputs, outputs, fname)


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
                        help="generate G & F models' dataset")

    parser.add_argument("--operator-noise", dest="operator_noise",
                        required=False,
                        action="store_true",
                        help="generate operators' dataset")

    parser.add_argument("--play-noise", dest="play_noise",
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
