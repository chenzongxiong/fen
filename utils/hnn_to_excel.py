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
    mu = 0
    sigma = 0
    points = 1000

    __activation__LIST = ['tanh', 'elu']
    __units__LIST = [1, 2, 8]
    __nb_plays__LIST = [1, 2, 8]

    nb_plays_LIST = [1]
    lr = 0.05
    epochs = 1000

    overview = []
    split_ratio = 0.6
    if argv.diff_weights:
        excel_fname = './new-dataset/models/diff_weights/method-sin/hnn-all.xlsx'
    else:
        excel_fname = './new-dataset/models/method-sin/hnn-all.xlsx'

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

        for __activation__ in __activation__LIST:
            for (__nb_plays__, __units__) in zip(__nb_plays__LIST, __units__LIST):

                if argv.diff_weights:
                    print("Loading from diff weights...")
                    prediction_fname = constants.DATASET_PATH['models_diff_weights_predictions'].format(method=method,
                                                                                                        activation=activation,
                                                                                                        state=state,
                                                                                                        mu=mu,
                                                                                                        sigma=sigma,
                                                                                                        units=units,
                                                                                                        nb_plays=nb_plays,
                                                                                                        points=points,
                                                                                                        input_dim=input_dim,
                                                                                                        __activation__=__activation__,
                                                                                                        __state__=0,
                                                                                                        __units__=__units__,
                                                                                                        __nb_plays__=__nb_plays__,
                                                                                                        loss='mse')
                    loss_file_fname = constants.DATASET_PATH['models_diff_weights_loss_history'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim,
                                                                                                        __activation__=__activation__, __state__=0, __units__=__units__, __nb_plays__=__nb_plays__, loss='mse')
                else:
                    prediction_fname = constants.DATASET_PATH['models_predictions'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim,
                                                                                           __activation__=__activation__, __state__=0, __units__=__units__, __nb_plays__=__nb_plays__, loss='mse')
                    loss_file_fname = constants.DATASET_PATH['models_loss_history'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim,
                                                                                           __activation__=__activation__, __state__=0, __units__=__units__, __nb_plays__=__nb_plays__, loss='mse')

                LOG.debug(colors.cyan("input file: {}".format(input_fname)))
                LOG.debug(colors.cyan("predict file: {}".format(prediction_fname)))
                LOG.debug(colors.cyan("loss file: {}".format(loss_file_fname)))
                predict_column = 'nb_plays-{}-units-{}-__activation__-{}-__nb_plays__-{}-__units__-{}-predictions'.format(nb_plays, units, __activation__, __nb_plays__, __units__)
                prediction = pd.read_csv(prediction_fname, header=None, names=['inputs', predict_column])
                loss_list = pd.read_csv(loss_file_fname, header=None, names=['loss'])


                kwargs = {predict_column: prediction[predict_column]}
                dataframe = dataframe.assign(**kwargs)
                loss = ((prediction[predict_column]  - base['outputs']).values ** 2).mean()**0.5
                kwargs = {"nb_plays-{}-units-{}-__activation__-__nb_plays__-{}-__units__-{}-loss".format(nb_plays, units, __activation__, __nb_plays__, __units__): loss}
                lossframe = lossframe.assign(**kwargs)

                overview.append([activation, __activation__, units, __nb_plays__, __units__, lr, epochs, loss_list['loss'].values[-1]])

        dataframe.to_excel(writer, sheet_name="nb_plays-{}-pred".format(nb_plays, index=False))
        lossframe.to_excel(writer, sheet_name="nb_plays-{}-loss".format(nb_plays, index=False))


    overview = pd.DataFrame(overview,
                            columns=['activation', '__activation__', 'nb_plays/units', '__nb_plays__', '__units__', 'adam_learning_rate', 'epochs', 'mse'])

    overview.to_excel(writer, sheet_name='overview', index=False)
    writer.close()
