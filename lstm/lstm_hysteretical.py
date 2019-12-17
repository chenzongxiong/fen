import sys

sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..")


import os
import argparse
import time
import tensorflow as tf
import numpy as np

import log as logging
import trading_data as tdata
import constants
import colors

LOG = logging.getLogger(__name__)

# input vs. output
def lstm(input_fname, units, epochs=1000, weights_fname=None, force_train=False, learning_rate=0.001):

    _train_inputs, _train_outputs = tdata.DatasetLoader.load_train_data(input_fname)
    _test_inputs, _test_outputs = tdata.DatasetLoader.load_test_data(input_fname)

    train_inputs = _train_inputs.reshape(-1, 1, 1)
    train_outputs = _train_outputs.reshape(-1, 1, 1)
    # learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    start = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(units),
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    if force_train or not os.path.isfile(weights_fname) :
        model.fit(train_inputs, train_outputs, epochs=epochs, verbose=1,
                  # callbacks=[early_stopping_callback],
                  validation_split=0.05,
                  shuffle=False)
        os.makedirs(os.path.dirname(weights_fname), exist_ok=True)
        model.save_weights(weights_fname)
    else:
        model.load_weights(weights_fname)

    end = time.time()
    LOG.debug(colors.red("time cost: {}s".format(end- start)))


    test_inputs = _test_inputs.reshape(-1, 1, 1)
    test_outputs = _test_outputs.reshape(-1)

    predictions = model.predict(test_inputs)
    pred_outputs = predictions.reshape(-1)
    rmse = np.sqrt(np.mean((pred_outputs - test_outputs) ** 2))

    LOG.debug(colors.red("LSTM rmse: {}".format(rmse)))

    return _test_inputs, pred_outputs, rmse, end-start


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs",
                        required=False, default=100,
                        type=int)
    parser.add_argument('--activation', dest='activation',
                        required=False,
                        default=None,
                        help='acitvation of non-linear layer')
    parser.add_argument("--mu", dest="mu",
                        required=False,
                        type=float)
    parser.add_argument("--sigma", dest="sigma",
                        required=False,
                        type=float)
    parser.add_argument("--lr", dest="lr",
                        required=False, default=0.001,
                        type=float)
    parser.add_argument("--points", dest="points",
                        required=False,
                        type=int)
    parser.add_argument("--nb_plays", dest="nb_plays",
                        required=False,
                        type=int)
    parser.add_argument("--units", dest="units",
                        required=False,
                        type=int)
    parser.add_argument("--__units__", dest="__units__",
                        required=False,
                        type=int)
    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")
    parser.add_argument('--force_train', dest='force_train',
                        required=False,
                        action="store_true")

    argv = parser.parse_args(sys.argv[1:])

    if argv.diff_weights is True:
        file_key = 'models_diff_weights'
    else:
        file_key = 'models'

    activation = argv.activation
    nb_plays = argv.nb_plays
    units = argv.units
    __units__ = argv.__units__  # 16, 32, 64, 128
    mu = int(argv.mu)
    sigma = int(argv.sigma)
    points = argv.points
    epochs = argv.epochs
    force_train = argv.force_train
    lr = argv.lr
    state = 0
    method = 'sin'
    input_dim = 1

    input_fname = constants.DATASET_PATH[file_key].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)

    LOG.debug("====================INFO====================")
    LOG.debug(colors.cyan("units: {}".format(units)))
    LOG.debug(colors.cyan("__units__: {}".format(__units__)))
    # LOG.debug(colors.cyan("method: {}".format(method)))
    LOG.debug(colors.cyan("nb_plays: {}".format(nb_plays)))
    # LOG.debug(colors.cyan("input_dim: {}".format(input_dim)))
    # LOG.debug(colors.cyan("state: {}".format(state)))
    LOG.debug(colors.cyan("mu: {}".format(mu)))
    LOG.debug(colors.cyan("sigma: {}".format(sigma)))
    LOG.debug(colors.cyan("activation: {}".format(activation)))
    LOG.debug(colors.cyan("points: {}".format(points)))
    LOG.debug(colors.cyan("epochs: {}".format(epochs)))
    LOG.debug(colors.cyan("lr: {}".format(lr)))
    LOG.debug(colors.cyan("Write  data to file {}".format(input_fname)))
    LOG.debug("================================================================================")

    prediction_fname = constants.DATASET_PATH['lstm_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
    loss_fname = constants.DATASET_PATH['lstm_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)
    weights_fname = constants.DATASET_PATH['lstm_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__)


    test_inputs, predictions, rmse, diff_tick = lstm(input_fname, units=__units__,
                                                     epochs=epochs, weights_fname=weights_fname,
                                                     force_train=force_train,
                                                     learning_rate=lr)

    tdata.DatasetSaver.save_data(test_inputs, predictions, prediction_fname)
    tdata.DatasetSaver.save_loss({"rmse": float(rmse), "diff_tick": float(diff_tick)}, loss_fname)
