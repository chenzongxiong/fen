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
_nb_plays = constants.NB_PLAYS
points = constants.POINTS


def operator_generator():
    states = [0, 1, 4, 7, 10 -1, -4, -7, -10]

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
                fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
                inputs = np.hstack(inputs)
                outputs = np.hstack(outputs)
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
                fname = "./training-data/plays/{}-{}-{}-tanh.csv".format(method, weight, width)
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

                for nb_plays in _nb_plays:
                    inputs, outputs, plays_outputs = tdata.DatasetGenerator.systhesis_model_generator(
                        nb_plays=nb_plays, points=points, debug_plays=True, inputs=inputs)

                    fname = "./training-data/models/{}-{}-{}-{}.csv".format(method, weight, width, nb_plays)
                    tdata.DatasetSaver.save_data(inputs, outputs, fname)
                    fname = "./training-data/models/{}-{}-{}-{}-multi.csv".format(method, weight, width, nb_plays)
                    tdata.DatasetSaver.save_data(inputs, plays_outputs, fname)


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


    argv = parser.parse_args(sys.argv[1:])

    if argv.operator:
        operator_generator()
    if argv.play:
        play_generator()
    if argv.model:
        model_generator()
