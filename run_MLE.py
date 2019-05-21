import sys
import time
import numpy as np

import log as logging
from core import MyModel, confusion_matrix
import utils
import trading_data as tdata
import constants
import colors

sess = utils.get_session()
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
        loss_name='mse'):

    epochs = 500
    # steps_per_epoch = batch_size

    start = time.time()
    input_dim = 10
    timestep = inputs.shape[0] // input_dim
    steps_per_epoch = input_dim
    # steps_per_epoch = 1

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      learning_rate=learning_rate)
    LOG.debug("Learning rate is {}".format(learning_rate))
    # mymodel.load_weights(weights_fname)
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
                     loss_file_name=loss_file_name)

    else:
        raise Exception("loss {} not support".format(loss_name))
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    # mymodel.weights
    mymodel.save_weights(weights_fname)

    predictions, mu, sigma = mymodel.predict2(inputs)

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
    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()
    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    start = time.time()
    predictions_list = []

    input_dim = shape[2]
    timestep = shape[1]
    num_samples = inputs.shape[0] // (input_dim * timestep)

    start = time.time()
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)

    mymodel.load_weights(weights_fname)
    for i in range(num_samples):
        LOG.debug("Predict on #{} sample".format(i+1))
        pred, mu, sigma = mymodel.predict2(inputs[i*(input_dim*timestep): (i+1)*(input_dim*timestep)])
        predictions_list.append(pred)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))

    predictions = np.hstack(predictions_list)
    outputs = outputs[:predictions.shape[-1]]
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))

    return predictions, loss


def trend(prices,
          B,
          mu,
          sigma,
          units=1,
          activation='tanh',
          nb_plays=1,
          weights_name='model.h5',
          trends_list_fname=None):

    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()
    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]
    timestep = prices.shape[0] // input_dim
    shape[1] = timestep

    start = time.time()

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)


    mymodel.load_weights(weights_fname, extra={'shape': shape})

    guess_trend, guess_trend_list = mymodel.trend(prices=prices, B=B, mu=mu, sigma=sigma)
    # loss = ((prediction - prices) ** 2).mean()
    # loss = float(loss)
    # LOG.debug("loss: {}".format(loss))
    # import ipdb; ipdb.set_trace()
    if trends_list_fname is not None:
        inputs = prices[1000:1000+guess_trend.shape[-1]]
        outputs = guess_trend_list
        tdata.DatasetSaver.save_data(inputs, outputs, trends_list_fname)

    loss = float(-1.0)
    return guess_trend, loss

def plot(a, b, trend_list):
    from matplotlib import pyplot as plt
    x = range(1, a.shape[0]+1)
    diff1 = ((a[1:] - a[:-1]) >= 0).tolist()
    diff2 = ((b[1:] - a[:-1]) >= 0).tolist()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, a, color='blue')
    ax1.plot(x, b, color='black')
    markers = ['^', 's']
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
    import ipdb; ipdb.set_trace()
    ax3.plot(x, a, color='blue')
    trend_list_ = [trend for trend in  trend_list]
    ax3.boxplot(trend_list_)

    plt.show()
    fname = "/Users/baymax_testios/Desktop/1.png"
    fig.savefig(fname, dpi=400)


if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    # Hyper Parameters
    # learning_rate = 0.003
    learning_rate = 0.01

    loss_name = 'mse'
    loss_name = 'mle'

    method = 'sin'
    # method = 'mixed'
    # method = 'noise'
    interp = 1
    do_prediction = False
    do_trend = False
    do_confusion_matrix = True
    mc_mode = True
    do_trend = False

    with_noise = True

    diff_weights = True

    run_test = False

    mu = 0
    sigma = 50

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
    __nb_plays__ = 50
    __units__ = 100
    # __nb_plays__ = 50
    # __units__ = 50

    __state__ = 0
    __activation__ = 'tanh'
    __activation__ = 'relu'
    # __activation__ = None
    __mu__ = 0
    __sigma__ = 70
    # __sigma__ = 5
    # __sigma__ = 20
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

    # weights_file_key = 'models_diff_weights_mc_stock_model_saved_weights'

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
                                                                    loss=loss_name)
    # method = 'noise'
    # sigma = 0.5

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


    inputs, outputs= tdata.DatasetLoader.load_data(fname)
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
    import ipdb; ipdb.set_trace()
    # inputs, outputs = outputs, inputs
    # inputs, outputs = inputs[:1100], outputs[:1100]
    import ipdb; ipdb.set_trace()

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
                                                                     loss=loss_name)

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
                                                                          loss=loss_name)

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
                                                                                loss=loss_name)

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
        import ipdb; ipdb.set_trace()
        predictions, loss = trend(prices=inputs,
                                  B=outputs,
                                  mu=__mu__,
                                  sigma=__sigma__,
                                  units=__units__,
                                  activation=__activation__,
                                  nb_plays=__nb_plays__,
                                  weights_name=weights_fname,
                                  trends_list_fname=trends_list_fname)
        inputs = inputs[1000:1000+predictions.shape[-1]]
    elif do_prediction is True:
        LOG.debug(colors.red("Load weights from {}".format(weights_fname)))
        import ipdb; ipdb.set_trace()
        predictions, loss = predict(inputs=inputs,
                                    outputs=outputs,
                                    units=__units__,
                                    activation=__activation__,
                                    nb_plays=__nb_plays__,
                                    weights_name=weights_fname)
    else:
        LOG.debug("START to FIT via {}".format(colors.red(loss_name.upper())))
        import ipdb; ipdb.set_trace()
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
                                loss_name=loss_name)

    LOG.debug("Write data into predicted_fname: {}".format(predicted_fname))
    tdata.DatasetSaver.save_data(inputs, predictions, predicted_fname)
