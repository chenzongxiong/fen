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
        import ipdb; ipdb.set_trace()
        # tensorflow case
        if isinstance(inputs, np.ndarray):
            num_sequence = inputs.shape[0]
        # keras case, inputs.shape -> (None, 1200)
        elif isinstance(inputs, tf.Tensor):
            num_sequence = inputs.shape[-1].value

        # num_batches = num_sequence // self._batch_size
        # batches = [(i*self._batch_size, (i+1)*self._batch_size) for i in range(num_batches)]

        # outputs = []
        # for play in self._plays:
        #     output = []
        #     for batch_start, batch_end in batches:
        #         if len(inputs.shape) == 1:
        #             output.append(play(inputs[batch_start:batch_end]))
        #         elif len(inputs.shape) == 2:
        #             output.append(play(inputs[:, batch_start:batch_end]))
        #         else:
        #             raise
        #     outputs.append(output)
        outputs = []
        for play in self._plays:
            outputs.append(play(inputs))

        import ipdb; ipdb.set_trace()
        outputs = tf.convert_to_tensor(outputs, dtype=self.dtype)   # (nb_plays, nb_batches, batch_size)

        # assert outputs.shape[0].value == self._nb_plays

        # outputs = tf.reshape(outptus, shape=(outputs.shape[1].value,
        #                                      outputs.shape[2].value))
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
    inputs, outputs = utils.load(fname)
    inputs = inputs.reshape(1, inputs.shape[0])
    outputs = outputs.reshape(1, outputs.shape[0])

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    batch_size = 600
    nb_players = 1
    play_model = PlayModel(nb_players, batch_size)

    play_model.compile(loss="mse",
                       optimizer=optimizer,
                       metrics=["mse"])

    import ipdb; ipdb.set_trace()
    LOG.debug("inputs.shape: {}, outputs.shape: {}".format(inputs.shape, outputs.shape))
    LOG.debug("Fitting...")
    play_model.fit(inputs, outputs, epochs=500, verbose=0, batch_size=batch_size,
                   shuffle=False)

    import ipdb; ipdb.set_trace()
    LOG.debug("Evaluating...")
    loss, mse = play_model.evaluate(inputs, outputs, verbose=0, batch_size=batch_size)
