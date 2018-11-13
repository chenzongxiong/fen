import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from play import Play, PlayCell

import utils
import log as logging

LOG = logging.getLogger(__name__)
SESS = tf.Session()


class PlayModel(tf.keras.Model):
    def __init__(self, nb_plays, units=4, batch_size=1):
        super(PlayModel, self).__init__(name="play_model")

        self._nb_plays = nb_plays
        self._plays = []
        self._batch_size = batch_size

        for _ in range(self._nb_plays):
            cell = PlayCell(debug=False)
            play = Play(units=units, cell=cell, debug=False)
            self._plays.append(play)
        self.plays_outputs = None

    def call(self, inputs, debug=False):
        """
        Parameters:
        ----------------
        inputs: `inputs` is a vector, assert len(inputs.shape) == 1
        """
        # import ipdb; ipdb.set_trace()
        outputs = []
        for play in self._plays:
            outputs.append(play(inputs))

        outputs = tf.convert_to_tensor(outputs, dtype=self.dtype)   # (nb_plays, nb_batches, batch_size)
        self.plays_outputs = outputs
        outputs = tf.reduce_sum(outputs, axis=0)
        LOG.debug("outputs.shape: {}".format(outputs.shape))  # (nb_plays, nb_batches, batch_size)
        outputs = tf.reshape(outputs, shape=(-1, outputs.shape[0].value))
        if debug is True:
            return outputs, self.plays_outputs
        else:
            return outputs

    def get_config(self):
        config = {
            "nb_plays": self._nb_plays,
            "batch_size": self._batch_size}
        return config

    def get_plays_outputs(self, inputs, batch_size=1, sess=None):
        if not sess:
            sess = tf.keras.backend.get_session()

        assert len(inputs.shape) == 2
        samples, _ = inputs.shape

        plays_outputs_list = []
        for x in range(samples):
            outputs, plays_outputs = self.__call__(inputs[x,:], debug=True)
            outputs_eval = sess.run(outputs)
            plays_outputs_eval = sess.run(plays_outputs)
            plays_outputs_list.append(plays_outputs_eval)

        return np.hstack(plays_outputs_list).T


if __name__ == "__main__":
    method = "sin"
    weight = 2
    width = 5

    # fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    fname = "./training-data/players/{}-{}-{}-4-tanh.csv".format(method, weight, width)
    (train_inputs, train_outputs), (test_inputs, test_outputs) = utils.load_data(fname, split=0.6)

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
    epochs = 3000
    # epochs = 500
    play_model = PlayModel(nb_plays, batch_size)

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
