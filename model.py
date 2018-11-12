import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from play import Play, PlayCell

import utils
import log as logging

LOG = logging.getLogger(__name__)


class PlayModel(tf.keras.Model):
    def __init__(self, nb_plays, batch_size=32):
        super(PlayModel, self).__init__(name="play_model")

        self._nb_plays = nb_plays
        self._plays = []
        self._batch_size = batch_size
        units = 4
        for _ in range(self._nb_plays):
            cell = PlayCell(debug=False)
            play = Play(units=units, cell=cell, debug=False)
            self._plays.append(play)

    def call(self, inputs):
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
        outputs = tf.reduce_sum(outputs, axis=0)
        LOG.debug("outputs.shape: {}".format(outputs.shape))  # (nb_plays, nb_batches, batch_size)
        return outputs

    def get_config(self):
        config = {
            "nb_plays": self._nb_plays,
            "batch_size": self._batch_size}
        return config


if __name__ == "__main__":
    method = "sin"
    weight = 2
    width = 5

    fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    (train_inputs, train_outputs), (test_inputs, test_outputs) = utils.load_data(fname, split=0.6)

    samples_per_batch = 10

    train_samples = train_inputs.shape[0] // samples_per_batch
    train_inputs = train_inputs.reshape(train_samples, samples_per_batch)  # samples * sequences
    train_outputs = train_outputs.reshape(train_samples, samples_per_batch)  # samples * sequences

    test_samples = test_inputs.shape[0] // samples_per_batch
    test_inputs = test_inputs.reshape(test_samples, samples_per_batch)  # samples * sequences
    test_outputs = test_outputs.reshape(test_samples, samples_per_batch)  # samples * sequences

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # NOTE: trick here, always set batch_size to 1, then reshape the input sequence.
    batch_size = 1
    nb_players = 3
    play_model = PlayModel(nb_players, batch_size)

    play_model.compile(loss="mse",
                       optimizer=optimizer,
                       metrics=["mse"])

    LOG.debug("train_inputs.shape: {}, train_outputs.shape: {}".format(train_inputs.shape, train_outputs.shape))
    LOG.debug("Fitting...")
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
    play_model.fit(train_inputs, train_outputs, epochs=5000, verbose=1, batch_size=batch_size,
                   shuffle=False, callbacks=[early_stopping_callback])
    LOG.debug("Evaluating...")
    # loss, mse = play_model.evaluate(train_inputs, train_outputs, verbose=1, batch_size=batch_size)
    loss, mse = play_model.evaluate(test_inputs, test_outputs, verbose=1, batch_size=batch_size)
    LOG.info("loss: {}, mse: {}".format(loss, mse))
