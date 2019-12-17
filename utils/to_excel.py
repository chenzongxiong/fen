import sys
import json
sys.path.append('.')
sys.path.append('..')
import pandas as pd
import constants
import log as logging



LOG = logging.getLogger(__name__)

if __name__ == "__main__":

    method = 'sin'
    state = 0
    input_dim = 1

    activation = 'tanh'
    nb_plays = 1
    units = 1
    mu = 0
    sigma = 0
    points = 1000
    __units__ = 1

    file_key = 'models'

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
    __units__LIST = [1, 8, 16, 32, 64, 128, 256]
    nb_plays_LIST = [1, 50]
    lr = 0.005
    epochs = 1000

    # rmse_list = []
    # time_cost_list = []
    # units_list = []
    # lstm_units_list = []
    # epochs_list = []
    # adam_learning_rate_list = []
    overview = []
    split_ratio = 0.6

    lossframe = pd.DataFrame({})

    excel_fname = 'pandas_multiple.xlsx'

    writer = pd.ExcelWriter(excel_fname, engine='xlsxwriter')

    for nb_plays in nb_plays_LIST:
        units = nb_plays

        input_fname = constants.DATASET_PATH[file_key].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)
        base = pd.read_csv(input_fname, header=None, names=['inputs', 'outputs'], skiprows=int(0.6*points))

        dataframe = base.copy(deep=False)

        for __units__ in __units__LIST:

            prediction_fname = constants.DATASET_PATH['lstm_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
            loss_fname = constants.DATASET_PATH['lstm_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
            loss_file_fname = constants.DATASET_PATH['lstm_loss_file'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)

            predict_column = 'nb_plays-{}-units-{}-predictions'.format(nb_plays, __units__)
            prediction = pd.read_csv(prediction_fname, header=None, names=['inputs', predict_column])
            loss_list = pd.read_csv(loss_file_fname, header=None, names=['loss'])

            kwargs = {predict_column: prediction[predict_column]}
            dataframe = dataframe.assign(**kwargs)
            kwargs = {"nb_plays-{}-units-{}-loss".format(nb_plays, __units__): loss_list['loss']}
            lossframe = lossframe.assign(**kwargs)


            with open(loss_fname) as f:
                loss = json.loads(f.read())

            overview.append([units, __units__, lr, epochs, loss['rmse'], loss['diff_tick']])

        dataframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-pred".format(nb_plays, '1-256', index=False))
        lossframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-loss".format(nb_plays, '1-256', index=False))

    overview = pd.DataFrame(overview,
                            columns=['nb_plays/units', 'lstm_units', 'adam_learning_rate', 'epochs', 'rmse', 'time_cost_(s)'])

    overview.to_excel(writer, sheet_name='overview', index=False)
    writer.close()
