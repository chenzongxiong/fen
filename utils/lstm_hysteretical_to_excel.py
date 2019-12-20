import sys
import json
sys.path.append('.')
sys.path.append('..')

import argparse

import pandas as pd
import constants
import log as logging

import colors


LOG = logging.getLogger(__name__)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")
    argv = parser.parse_args(sys.argv[1:])


    method = 'sin'
    state = 0
    input_dim = 1

    activation = 'tanh'
    nb_plays = 1
    units = 1
    mu = 0
    sigma = 2
    points = 1000

    # LOG.debug("====================INFO====================")
    # LOG.debug(colors.cyan("units: {}".format(units)))
    # LOG.debug(colors.cyan("__units__: {}".format(__units__)))
    # # LOG.debug(colors.cyan("method: {}".format(method)))
    # LOG.debug(colors.cyan("nb_plays: {}".format(nb_plays)))
    # # LOG.debug(colors.cyan("input_dim: {}".format(input_dim)))
    # # LOG.debug(colors.cyan("state: {}".format(state)))
    # LOG.debug(colors.cyan("mu: {}".format(mu)))
    # LOG.debug(colors.cyan("sigma: {}".format(sigma)))
    # LOG.debug(colors.cyan("activation: {}".format(activation)))
    # LOG.debug(colors.cyan("points: {}".format(points)))
    # LOG.debug(colors.cyan("epochs: {}".format(epochs)))
    # LOG.debug(colors.cyan("Write  data to file {}".format(input_fname)))
    # LOG.debug("================================================================================")
    input(colors.red("RUN script ./lstm_loss_history_collector.sh #diff_weights before run this script"))
    __units__LIST = [1, 8, 16, 32, 64, 128, 256]
    nb_plays_LIST = [1, 50, 100, 500]
    lr = 0.001
    epochs = 1000

    overview = []
    split_ratio = 0.6
    if argv.diff_weights:
        excel_fname = './new-dataset/lstm/diff_weights/method-sin/lstm-all-sigma-2.xlsx'
    else:
        excel_fname = './new-dataset/lstm/method-sin/lstm-all-sigma-2.xlsx'

    writer = pd.ExcelWriter(excel_fname, engine='xlsxwriter')

    for nb_plays in nb_plays_LIST:
        lossframe = pd.DataFrame({})
        units = nb_plays
        if nb_plays == 500:
            units = 100


        if argv.diff_weights:
            input_fname = constants.DATASET_PATH['models_diff_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)
        else:
            input_fname = constants.DATASET_PATH['models'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)

        base = pd.read_csv(input_fname, header=None, names=['inputs', 'outputs'], skiprows=int(0.6*points))

        dataframe = base.copy(deep=False)

        for __units__ in __units__LIST:
            if argv.diff_weights:
                prediction_fname = constants.DATASET_PATH['lstm_diff_weights_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
                loss_fname = constants.DATASET_PATH['lstm_diff_weights_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
                loss_file_fname = constants.DATASET_PATH['lstm_diff_weights_loss_file'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, learning_rate=lr)
            else:
                prediction_fname = constants.DATASET_PATH['lstm_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
                loss_fname = constants.DATASET_PATH['lstm_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
                loss_file_fname = constants.DATASET_PATH['lstm_loss_file'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, learning_rate=lr)

            predict_column = 'nb_plays-{}-units-{}-predictions'.format(nb_plays, __units__)
            prediction = pd.read_csv(prediction_fname, header=None, names=['inputs', predict_column])
            loss_list = pd.read_csv(loss_file_fname, header=None, names=['loss'])

            kwargs = {predict_column: prediction[predict_column]}
            dataframe = dataframe.assign(**kwargs)
            kwargs = {"nb_plays-{}-units-{}-loss".format(nb_plays, __units__): loss_list['loss']}
            lossframe = lossframe.assign(**kwargs)

            with open(loss_fname) as f:
                loss = json.loads(f.read())

            rmse = ((base['outputs'] - prediction[predict_column]).values ** 2).mean() ** 0.5
            overview.append([nb_plays, __units__, lr, epochs, rmse, loss['diff_tick']])

        dataframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-pred".format(nb_plays, '1-256', index=False))
        lossframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-loss".format(nb_plays, '1-256', index=False))

    overview = pd.DataFrame(overview,
                            columns=['nb_plays/units', 'lstm_units', 'adam_learning_rate', 'epochs', 'rmse', 'time_cost_(s)'])

    overview.to_excel(writer, sheet_name='overview', index=False)
    writer.close()
