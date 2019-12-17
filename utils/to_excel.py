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

    input_fname = constants.DATASET_PATH[file_key].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)
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
    prediction_fname = constants.DATASET_PATH['lstm_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
    loss_fname = constants.DATASET_PATH['lstm_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)

    base = pd.read_csv(input_fname, header=None, names=['inputs', 'outputs'])
    prediction = pd.read_csv(prediction_fname, header=None, names=['inputs', 'predictions'])
    with open(loss_fname) as f:
        loss = json.loads(f.read())

    import ipdb; ipdb.set_trace()




    # weights_fname = constants.DATASET_PATH['lstm_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
