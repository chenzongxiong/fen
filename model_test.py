import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import core
import utils
import trading_data as tdata
import colors
import log as logging

LOG = logging.getLogger(__name__)


if __name__ == "__main__":
    method = "sin"
    weight = 2
    width = 5

    fname = "./training-data/plays/{}-{}-{}-4-tanh.csv".format(method, weight, width)
    train_inputs, train_outputs = tdata.DatasetLoader.load_train_data(fname)
    test_inputs, test_outputs = tdata.DatasetLoader.load_test_data(fname)
    samples_per_batch = 240
    # samples_per_batch = 10

    train_samples = train_inputs.shape[0] // samples_per_batch
    train_inputs = train_inputs.reshape(train_samples, samples_per_batch)  # samples * sequences
    train_outputs = train_outputs.reshape(train_samples, samples_per_batch)  # samples * sequences

    test_samples = test_inputs.shape[0] // samples_per_batch
    test_inputs = test_inputs.reshape(test_samples, samples_per_batch)  # samples * sequences
    test_outputs = test_outputs.reshape(test_samples, samples_per_batch)  # samples * sequences

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # NOTE: trick here, always set batch_size to 1, then reshape the input sequence.
    batch_size = 1
    nb_plays = 3
    epochs = 1000
    # epochs = 500
    play_model = core.PlayModel(nb_plays, batch_size)

    play_model.compile(loss="mse",
                       optimizer=optimizer,
                       metrics=["mse"])

    LOG.debug("train_inputs.shape: {}, train_outputs.shape: {}".format(train_inputs.shape, train_outputs.shape))
    LOG.debug("Fitting...")
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    play_model.fit(train_inputs, train_outputs, epochs=epochs, verbose=1, batch_size=batch_size,
                   shuffle=False, callbacks=[early_stopping_callback])

    LOG.debug("Evaluating...")
    loss, mse = play_model.evaluate(train_inputs, train_outputs, verbose=1, batch_size=batch_size)
    loss, mse = play_model.evaluate(test_inputs, test_outputs, verbose=1, batch_size=batch_size)
    LOG.info("loss: {}, mse: {}".format(loss, mse))

    train_predications = play_model.predict(train_inputs, batch_size=batch_size, verbose=1)
    # LOG.debug("predications of train_inputs: {}".format(predications))
    test_predications = play_model.predict(test_inputs, batch_size=batch_size, verbose=1)
    # LOG.debug("predications of test_inputs: {}".format(predications))

    train_plays_outputs = play_model.get_plays_outputs(train_inputs)
    test_plays_outputs = play_model.get_plays_outputs(test_inputs)

    train_inputs = train_inputs.reshape(train_samples*samples_per_batch)
    train_outputs = train_outputs.reshape(train_samples*samples_per_batch)
    train_predications = train_predications.reshape(train_samples*samples_per_batch)
    # train_plays_outputs = train_plays_outputs.T

    test_inputs = test_inputs.reshape(test_samples*samples_per_batch)
    test_outputs = test_outputs.reshape(test_samples*samples_per_batch)
    test_predications = test_predications.reshape(test_samples*samples_per_batch)
    # test_plays_outputs = test_plays_outputs.T

    utils.save_data(train_inputs, train_predications, "train1.csv")
    utils.save_data(test_inputs, test_predications, "test1.csv")

    utils.save_data(train_inputs, train_plays_outputs, "train2.csv")
    utils.save_data(test_inputs, test_plays_outputs, "test2.csv")

    inputs, outputs = utils.load_data(fname)
    _, train_predications = utils.load_data("./train1.csv")
    _, test_predications = utils.load_data("./test1.csv")
    _, train_plays_outputs = utils.load_data("./train2.csv")
    _, test_plays_outputs = utils.load_data("./test2.csv")

    predicates = np.hstack([train_predications, test_predications])
    plays_outputs = np.hstack([train_plays_outputs, test_plays_outputs]).T

    if len(plays_outputs.shape) == 1:
        comb_outputs = np.vstack([outputs, predicates, plays_outputs]).T
    else:
        comb_outputs = np.vstack([outputs, predicates]).T
        comb_outputs = np.hstack([comb_outputs, plays_outputs])

    nbr_of_inputs = comb_outputs.shape[1]

    comb_inputs = np.vstack([inputs for _ in range(nbr_of_inputs)]).T

    anim_fname = "play.gif"
    utils.save_animation(comb_inputs, comb_outputs, anim_fname, colors=utils.generate_colors(comb_inputs.shape[1]), step=5)
