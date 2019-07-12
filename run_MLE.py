import os
import sys
import argparse

import time
import numpy as np

import log as logging
from core import MyModel, confusion_matrix
import trading_data as tdata
import constants
import colors
import utils
import tensorflow as tf

LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS


def fit(inputs,
        outputs,
        mu,
        sigma,
        units=1,
        activation='tanh',
        nb_plays=1,
        learning_rate=0.001,
        loss_file_name="./tmp/my_model_loss_history.csv",
        weights_name='model.h5',
        loss_name='mse',
        batch_size=10,
        ensemble=1):

    epochs = 20000
    # epochs = 1

    start = time.time()
    input_dim = batch_size

    # timestep = 1
    # input_dim = 1000
    timestep = inputs.shape[0] // input_dim
    # steps_per_epoch = inputs.shape[0] // input_dim
    steps_per_epoch = 1

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      learning_rate=learning_rate,
                      ensemble=ensemble)

    LOG.debug("Learning rate is {}".format(learning_rate))

    preload_weights = False

    if loss_name == 'mse':
        mymodel.fit(inputs,
                    outputs,
                    verbose=1,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    loss_file_name=loss_file_name,
                    learning_rate=learning_rate)
    elif loss_name == 'mle':
        mymodel.fit2(inputs=inputs,
                     mu=mu,
                     sigma=sigma,
                     outputs=outputs,
                     epochs=epochs,
                     verbose=1,
                     steps_per_epoch=steps_per_epoch,
                     loss_file_name=loss_file_name,
                     preload_weights=preload_weights,
                     weights_fname=weights_fname)

    else:
        raise Exception("loss {} not support".format(loss_name))
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    mymodel.save_weights(weights_fname)
    start = time.time()
    predictions = mymodel.predict(inputs)
    end = time.time()
    LOG.debug("Time cost in prediction: {}s".format(end-start))
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss


def predict(inputs,
            outputs,
            units=1,
            activation='tanh',
            nb_plays=1,
            weights_name='model.h5'):
    best_epoch = None
    try:
        with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
            line = f.read()
    except FileNotFoundError:
        epochs = []
        base = '/'.join(weights_fname.split('/')[:-1])
        for _dir in os.listdir(base):
            if os.path.isdir('{}/{}'.format(base, _dir)):
                try:
                    epochs.append(int(_dir.split('-')[-1]))
                except ValueError:
                    pass

        if not epochs:
            raise Exception("no trained parameters found")

        best_epoch = max(epochs)

        LOG.debug("Best epoch is {}".format(best_epoch))
        dirname = '{}-epochs-{}/{}plays'.format(weights_fname[:-3], best_epoch, nb_plays)
        if not os.path.isdir(dirname):
            # sanity checking
            raise Exception("Bugs inside *save_wegihts* or *fit2*")
        with open("{}/input_shape.txt".format(dirname), 'r') as f:
            line = f.read()

    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    start = time.time()
    predictions_list = []

    input_dim = shape[2]
    timestep = shape[1]
    num_samples = inputs.shape[0] // (input_dim * timestep)

    start = time.time()
    parallel_prediction = True
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=parallel_prediction)

    if parallel_prediction is True:
        mymodel.load_weights(weights_fname, extra={'shape': shape, 'parallelism': True})
        predictions = mymodel.predict_parallel(inputs)
    else:
        mymodel.load_weights(weights_fname)
        predictions, mu, sigma = mymodel.predict(inputs)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    outputs = outputs[:predictions.shape[-1]]
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))

    # return predictions, loss
    return predictions, best_epoch

def trend(prices,
          B,
          mu,
          sigma,
          units=1,
          activation='tanh',
          nb_plays=1,
          weights_name='model.h5',
          trends_list_fname=None):

    best_epoch = None
    # try:
    #     with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
    #         line = f.read()
    # except FileNotFoundError:
    if True:
        epochs = []
        base = '/'.join(weights_fname.split('/')[:-1])
        for _dir in os.listdir(base):
            if os.path.isdir('{}/{}'.format(base, _dir)):
                try:
                    epochs.append(int(_dir.split('-')[-1]))
                except ValueError:
                    pass

        if not epochs:
            raise Exception("no trained parameters found")

        best_epoch = max(epochs)
        best_epoch = 15000
        LOG.debug("Best epoch is {}".format(best_epoch))
        dirname = '{}-epochs-{}/{}plays'.format(weights_fname[:-3], best_epoch, nb_plays)
        if not os.path.isdir(dirname):
            # sanity checking
            raise Exception("Bugs inside *save_wegihts* or *fit2*")
        with open("{}/input_shape.txt".format(dirname), 'r') as f:
            line = f.read()

    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]

    timestep = 1
    shape[1] = timestep

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=True)

    mymodel.load_weights(weights_fname, extra={'shape': shape, 'parallelism': True, 'use_epochs': True, 'best_epoch': best_epoch})
    guess_trend = mymodel.trend(prices=prices, B=B, mu=mu, sigma=sigma)

    loss = float(-1.0)
    return guess_trend, loss


def plot_graphs_together(price_list, noise_list, mu, sigma,
                         units=1,
                         activation='tanh',
                         nb_plays=1,
                         weights_name='model.h5',
                         trends_list_fname=None, ensemble=1):
    best_epoch = None

    # try:
    #     with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
    #         line = f.read()
    # except FileNotFoundError:
    if True:
        epochs = []
        base = '/'.join(weights_fname.split('/')[:-1])
        for _dir in os.listdir(base):
            if os.path.isdir('{}/{}'.format(base, _dir)):
                try:
                    epochs.append(int(_dir.split('-')[-1]))
                except ValueError:
                    pass

        if not epochs:
            raise Exception("no trained parameters found")

        best_epoch = max(epochs)
        best_epoch = 15000
        LOG.debug("Best epoch is {}".format(best_epoch))
        dirname = '{}-epochs-{}/{}plays'.format(weights_fname[:-3], best_epoch, nb_plays)
        if not os.path.isdir(dirname):
            # sanity checking
            raise Exception("Bugs inside *save_wegihts* or *fit2*")
        with open("{}/input_shape.txt".format(dirname), 'r') as f:
            line = f.read()

    shape = list(map(int, line.split(":")))
    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]

    timestep = 1
    shape[1] = timestep
    parallelism = True
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=parallelism,
                      ensemble=ensemble)

    mymodel.load_weights(weights_fname, extra={'shape': shape, 'parallelism': parallelism, 'best_epoch': best_epoch, 'use_epochs': True})
    mymodel.plot_graphs_together(prices=price_list, noises=noise_list, mu=mu, sigma=sigma)


def visualize(inputs,
              mu=0,
              sigma=1,
              units=1,
              activation='tanh',
              nb_plays=1,
              weights_name='model.h5'):

    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()
    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]
    # timestep = inputs.shape[0] // input_dim
    timestep = 1
    shape[1] = timestep
    start = time.time()

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)

    mymodel.load_weights(weights_fname, extra={'shape': shape})
    mymodel.visualize_activated_plays(inputs=inputs)


def plot(a, b, trend_list):
    from matplotlib import pyplot as plt
    x = range(1, a.shape[0]+1)
    diff1 = ((a[1:] - a[:-1]) >= 0).tolist()
    diff2 = ((b[1:] - a[:-1]) >= 0).tolist()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, a, color='blue')
    ax1.plot(x, b, color='black')

    for index, d1, d2 in zip(x[1:], diff1, diff2):
        if d1 is True and d2 is True:
            ax1.scatter([index], [b[index-1]], marker='^', color='green')
        elif d1 is False and d2 is False:
            ax1.scatter([index], [b[index-1]], marker='^', color='green')
        elif d1 is False and d2 is True:
            ax1.scatter([index], [b[index-1]], marker='s', color='black')
        elif d1 is True and d2 is False:
            ax1.scatter([index], [b[index-1]], marker='s', color='black')

    ax2.plot(x, a, color='blue')
    min_trend_list = trend_list.min(axis=1)
    max_trend_list = trend_list.max(axis=1)
    ax2.fill_between(x, min_trend_list, max_trend_list, facecolor='gray', alpha=0.5, interpolate=True)
    ax3.plot(x, a, color='blue')
    trend_list_ = [trend for trend in  trend_list]
    ax3.boxplot(trend_list_)

    plt.show()
    fname = "/Users/baymax_testios/Desktop/1.png"
    fig.savefig(fname, dpi=400)


def ttest_rel(method1, method2):
    # outputs = np.array(outputs).reshape(-1)
    # guess_prices = np.array(guess_prices).reshape(-1)

    # loss1 =  ((guess_prices - prices[start_pos:end_pos]) ** 2)
    # loss2 = np.abs(guess_prices - prices[start_pos:end_pos])
    # loss3 = (prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1]) ** 2
    # loss4 = np.abs(prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1])

    # LOG.debug("root sum square loss1: {}".format((loss1.sum()/(end_pos-start_pos))**(0.5)))
    # LOG.debug("root sum square loss2: {}".format((loss3.sum()/(end_pos-start_pos))**(0.5)))
    # LOG.debug("total abs loss1: {}".format((loss2.sum()/(end_pos-start_pos))))
    # LOG.debug("total abs loss2: {}".format((loss4.sum()/(end_pos-start_pos))))

    # guess_prices_list = np.array(guess_prices_list)
    pass


if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size",
                        default=1000,
                        type=int)
    parser.add_argument("--__nb_plays__", dest="__nb_plays__",
                        default=2,
                        type=int)
    parser.add_argument("--__units__", dest="__units__",
                        default=5,
                        type=int)
    parser.add_argument("--__activation__", dest="__activation__",
                        default=None,
                        type=str)

    parser.add_argument('--trend', dest='trend',
                        default=False, action='store_true')
    parser.add_argument('--predict', dest='predict',
                        default=False, action='store_true')
    parser.add_argument('--plot', dest='plot',
                        default=False, action='store_true')
    parser.add_argument('--visualize_activated_plays', dest='visualize_activated_plays',
                        default=False, action='store_true')
    parser.add_argument('--__mu__', dest='__mu__',
                        default=0,
                        type=float)
    parser.add_argument('--__sigma__', dest='__sigma__',
                        default=110,
                        type=float)
    parser.add_argument('--ensemble', dest='ensemble',
                        default=2,  # start from 1
                        type=int)

    argv = parser.parse_args(sys.argv[1:])
    # Hyper Parameters
    # learning_rate = 0.003
    learning_rate = 0.07

    batch_size = argv.batch_size

    loss_name = 'mse'
    loss_name = 'mle'

    method = 'sin'
    # method = 'mixed'
    # method = 'noise'
    interp = 1
    # do_prediction = False
    do_prediction = argv.predict
    do_confusion_matrix = True
    mc_mode = True
    do_trend = argv.trend
    do_plot = argv.plot
    do_visualize_activated_plays = argv.visualize_activated_plays
    ensemble = argv.ensemble
    with_noise = True

    diff_weights = True

    run_test = False

    mu = 0
    sigma = 110

    points = 1000
    input_dim = 1
    ############################## ground truth #############################
    nb_plays = 20
    # units is 10000 special for dataset comes from simulation
    units = 20
    state = 0
    activation = 'tanh'
    activation = None
    ############################## predicitons #############################
    __nb_plays__ = argv.__nb_plays__
    __units__ = argv.__units__
    # __nb_plays__ = 2
    # __units__ = 2

    __state__ = 0
    __activation__ = argv.__activation__
    # __activation__ = 'relu'
    # __activation__ = None
    # __activation__ = 'tanh'
    # __mu__ = 2.60
    __mu__ = 0
    __sigma__ = 110
    # __mu__ = argv.__mu__
    # __sigma__ = argv.__sigma__

    if method == 'noise':
        with_noise = True

    if with_noise is False:
        mu = 0
        sigma = 0

    if diff_weights is True:
        # input_file_key = 'models_diff_weights'
        # loss_file_key = 'models_diff_weights_loss_history'
        if mc_mode is True:
            weights_file_key = 'models_diff_weights_mc_saved_weights'
        else:
            weights_file_key = 'models_diff_weights_saved_weights'
        # predictions_file_key = 'models_diff_weights_predictions'
        weights_file_key = 'models_diff_weights_mc_saved_weights'
    else:
        # input_file_key = 'models'
        # loss_file_key = 'models_loss_history'
        # weights_file_key = 'models_saved_weights'
        # predictions_file_key = 'models_predictions'
        raise

    weights_file_key = 'models_diff_weights_mc_stock_model_saved_weights'

    # XXXX: place weights_fname before run_test
    weights_fname = constants.DATASET_PATH[weights_file_key].format(method=method,
                                                                    activation=activation,
                                                                    state=state,
                                                                    mu=mu,
                                                                    sigma=sigma,
                                                                    units=units,
                                                                    nb_plays=nb_plays,
                                                                    points=points,
                                                                    input_dim=input_dim,
                                                                    __activation__=__activation__,
                                                                    __state__=__state__,
                                                                    __units__=__units__,
                                                                    __nb_plays__=__nb_plays__,
                                                                    loss=loss_name,
                                                                    ensemble=ensemble,
                                                                    batch_size=batch_size)
    if interp != 1:
        if do_prediction is False:
            raise
        if run_test is True:
            raise
        elif run_test is False:
            raise
    elif interp == 1:
        if run_test is True:
            raise
        elif run_test is False:
            if diff_weights is True:
                input_file_key = 'models_diff_weights'
                loss_file_key = 'models_diff_weights_loss_history'
                predictions_file_key = 'models_diff_weights_predictions'
            else:
                raise

    # if do_trend is True:
    ################### markov chain #############################
    if mc_mode is True:
        input_file_key = 'models_diff_weights_mc_stock_model'
        loss_file_key = 'models_diff_weights_mc_stock_model_loss_history'
        predictions_file_key = 'models_diff_weights_mc_stock_model_predictions'
        if do_trend is True:
            predictions_file_key = 'models_diff_weights_mc_stock_model_trends'
            trends_list_file_key = 'models_diff_weights_mc_stock_model_trends_list'
    else:
        input_file_key = 'models_diff_weights_mc'
        loss_file_key = 'models_diff_weights_mc_loss_history'
        predictions_file_key = 'models_diff_weights_mc_predictions'

    fname = constants.DATASET_PATH[input_file_key].format(interp=interp,
                                                          method=method,
                                                          activation=activation,
                                                          state=state,
                                                          mu=mu,
                                                          sigma=sigma,
                                                          units=units,
                                                          nb_plays=nb_plays,
                                                          points=points,
                                                          input_dim=input_dim)
    LOG.debug("Load data from file: {}".format(colors.cyan(fname)))
    if do_prediction is True and do_trend is True:
        raise Exception("both do predictions and do_trend are True")


    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    if do_trend is False:
        inputs, outputs = inputs[:points], outputs[:points]
    if mc_mode is True:
        # inputs, outputs = outputs, inputs
        pass
    else:
        inputs, outputs = outputs, inputs
        # gap = 5
        # inputs, outputs = inputs[::gap], outputs[::gap]
        # # inputs = np.arange(800)[::4].astype(np.float32)
        # # inputs = np.zeros(800)[::4].astype(np.float32)
        # # mu = 0
        # # sigma = 0.5
        # # points = 200
        # # noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
        # # inputs += noise
        # mu1 = 4
        # sigma1 = 2.5
        # inputs = tdata.DatasetGenerator.systhesis_markov_chain_generator(200, mu1, sigma1)

        pass

    # inputs, outputs = outputs, inputs
    inputs, outputs = inputs[:2000], outputs[:2000]

    loss_history_file = constants.DATASET_PATH[loss_file_key].format(interp=interp,
                                                                     method=method,
                                                                     activation=activation,
                                                                     state=state,
                                                                     mu=mu,
                                                                     sigma=sigma,
                                                                     units=units,
                                                                     nb_plays=nb_plays,
                                                                     points=points,
                                                                     input_dim=input_dim,
                                                                     __activation__=__activation__,
                                                                     __state__=__state__,
                                                                     __units__=__units__,
                                                                     __nb_plays__=__nb_plays__,
                                                                     loss=loss_name,
                                                                     ensemble=ensemble,
                                                                     batch_size=batch_size)

    predicted_fname = constants.DATASET_PATH[predictions_file_key].format(interp=interp,
                                                                          method=method,
                                                                          activation=activation,
                                                                          state=state,
                                                                          mu=mu,
                                                                          sigma=sigma,
                                                                          units=units,
                                                                          nb_plays=nb_plays,
                                                                          points=points,
                                                                          input_dim=input_dim,
                                                                          __activation__=__activation__,
                                                                          __state__=__state__,
                                                                          __units__=__units__,
                                                                          __nb_plays__=__nb_plays__,
                                                                          loss=loss_name,
                                                                          ensemble=ensemble,
                                                                          batch_size=batch_size)

    if mc_mode is True and do_trend is True:
        trends_list_fname = constants.DATASET_PATH[trends_list_file_key].format(interp=interp,
                                                                                method=method,
                                                                                activation=activation,
                                                                                state=state,
                                                                                mu=mu,
                                                                                sigma=sigma,
                                                                                units=units,
                                                                                nb_plays=nb_plays,
                                                                                points=points,
                                                                                input_dim=input_dim,
                                                                                __activation__=__activation__,
                                                                                __state__=__state__,
                                                                                __units__=__units__,
                                                                                __nb_plays__=__nb_plays__,
                                                                                loss=loss_name,
                                                                                ensemble=ensemble,
                                                                                batch_size=batch_size)



    LOG.debug('############################  SETTINGS #########################################')
    LOG.debug('# Learning Rate: {}'.format(learning_rate))
    LOG.debug('# points: {}'.format(points))
    LOG.debug('# nb_plays: {}'.format(nb_plays))
    LOG.debug('# units: {}'.format(units))
    LOG.debug('# activation: {}'.format(activation))
    LOG.debug("# mu: {}".format(mu))
    LOG.debug("# sigma: {}".format(sigma))
    LOG.debug("# state: {}".format(state))
    LOG.debug('# __nb_plays__: {}'.format(__nb_plays__))
    LOG.debug('# __units__: {}'.format(__units__))
    LOG.debug('# __activation__: {}'.format(__activation__))
    LOG.debug("# __mu__: {}".format(__mu__))
    LOG.debug("# __sigma__: {}".format(__sigma__))
    LOG.debug("# __state__: {}".format(__state__))

    LOG.debug("# do_prediction: {}".format(do_prediction))
    LOG.debug("# do_trend: {}".format(do_trend))
    LOG.debug("# do_fit: {}".format(not (do_prediction and do_trend)))
    LOG.debug("# mc_mode: {}".format(mc_mode))

    LOG.debug('# train_fname: {}'.format(fname))
    LOG.debug('# predicted_fname: {}'.format(predicted_fname))
    LOG.debug('# weights_fname: {}'.format(weights_fname))
    LOG.debug('# weights_fname: {}'.format(weights_fname))
    LOG.debug('################################################################################')



    # try:
    #     import ipdb; ipdb.set_trace()
    #     a, b = tdata.DatasetLoader.load_data(predicted_fname)
    #     inp, trend_list = tdata.DatasetLoader.load_data(trends_list_fname)
    #     assert np.allclose(a, inp, atol=1e-5)
    #     confusion = confusion_matrix(a, b)
    #     LOG.debug(colors.purple("confusion matrix is: {}".format(confusion)))

    #     plot(a, b, trend_list)
    #     sys.exit(0)
    # except FileNotFoundError:
    #     LOG.warning("Not found prediction file, no way to create confusion matrix")

    if mc_mode is True and do_trend is True:
        predictions, loss = trend(prices=inputs[:batch_size*2],
                                  B=outputs[:batch_size*2],
                                  mu=__mu__,
                                  sigma=__sigma__,
                                  units=__units__,
                                  activation=__activation__,
                                  nb_plays=__nb_plays__,
                                  weights_name=weights_fname,
                                  trends_list_fname=trends_list_fname)
        inputs = inputs[batch_size:batch_size+predictions.shape[-1]]
    elif do_visualize_activated_plays is True:
        LOG.debug(colors.red("Load weights from {}, DO VISUALIZE ACTIVATED PLAYS".format(weights_fname)))
        visualize(inputs=inputs[:batch_size],
                  mu=__mu__,
                  sigma=__sigma__,
                  units=__units__,
                  activation=__activation__,
                  nb_plays=__nb_plays__,
                  weights_name=weights_fname)
        sys.exit(0)
    elif do_prediction is True:
        LOG.debug(colors.red("Load weights from {}".format(weights_fname)))
        # import ipdb; ipdb.set_trace()
        inputs, outputs = inputs[:batch_size], outputs[:batch_size]
        predictions, best_epoch = predict(inputs=inputs,
                                           outputs=outputs,
                                           units=__units__,
                                           activation=__activation__,
                                           nb_plays=__nb_plays__,
                                           weights_name=weights_fname)
        if best_epoch is not None:
            predicted_fname = "{}-epochs-{}.csv".format(predicted_fname[:-4], best_epoch)

    elif do_plot is True:
        inputs, outputs = inputs[:batch_size*2], outputs[:batch_size*2]
        plot_graphs_together(price_list=inputs, noise_list=outputs, mu=__mu__, sigma=__sigma__,
                             weights_name=weights_fname,
                             units=__units__,
                             activation=__activation__,
                             nb_plays=__nb_plays__,
                             ensemble=ensemble)
        sys.exit(0)
    else:
        LOG.debug("START to FIT via {}".format(colors.red(loss_name.upper())))
        inputs, outputs = inputs[:batch_size], outputs[:batch_size]

        predictions, loss = fit(inputs=inputs,
                                outputs=outputs,
                                mu=__mu__,
                                sigma=__sigma__,
                                units=__units__,
                                activation=__activation__,
                                nb_plays=__nb_plays__,
                                learning_rate=learning_rate,
                                loss_file_name=loss_history_file,
                                weights_name=weights_fname,
                                loss_name=loss_name,
                                batch_size=batch_size,
                                ensemble=ensemble)

    LOG.debug("Write data into predicted_fname: {}".format(predicted_fname))
    tdata.DatasetSaver.save_data(inputs, predictions, predicted_fname)
