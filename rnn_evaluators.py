import tensorflow as tf

# from tensorflow import keras

import trading_data as tdata
import constants
import log as logging


LOG = logging.getLogger(__name__)

methods = constants.METHODS
weights = constants.WEIGHTS
widths = constants.WIDTHS
nb_plays = constants.NB_PLAYS
epochs = constants.EPOCHS


class Evaluator(object):

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)

    @classmethod
    def lstm(cls, units):
        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.LSTM(units, input_shape=(units, 1)))
        cls.model.compile(loss='mse', optimizer=cls.optimizer, metrics=['mse'])
        return cls

    @classmethod
    def simple_rnn(cls, units):
        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.SimpleRNN(units, input_shape=(units, 1)))
        cls.model.compile(loss='mse', optimizer=cls.optimizer, metrics=['mse'])
        return cls


    @classmethod
    def gru(cls, units):
        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.GRU(units, input_shape=(units, 1)))
        cls.model.compile(loss='mse', optimizer=cls.optimizer, metrics=['mse'])
        return cls

    @classmethod
    def fit(cls, inputs, outputs):
        cls.model.fit(inputs, outputs, epochs=epochs,
                      verbose=1, callbacks=[cls.early_stopping_callback])
        return cls

    @classmethod
    def evaluate(cls, inputs, output):
        loss, mse = cls.model.evaluate(inputs, outputs, verbose=0, batch_size=1200)
        LOG.debug("loss: {}, mse: {}".format(loss, mse))
        return cls

    @classmethod
    def predict(cls, inputs, outputs):
        predictions = cls.model.predict(inputs)
        return predictions

if __name__ == "__main__":

    for method in methods:
        for weight in weights:
            for width in widths:
                for _nb_plays in nb_plays:
                    fname = constants.FNAME_FORMAT["models"].format(method=method, weight=weight, width=width, nb_plays=_nb_plays)
                    inputs, outputs = tdata.DatasetLoader.load_data(fname)
                    inputs = inputs.reshape(1, -1, 1)
                    outputs = outputs.reshape(1, -1)
                    units = outputs.shape[-1]
                    Evaluator.simple_rnn(units).fit(inputs, outputs).evaluate(inputs, outputs)
                    Evaluator.lstm(units).fit(inputs, outputs).evalute(inputs, outputs)
                    Evaluator.gru(units).fit(inputs, outputs).evaluate(inputs, outputs)
                    # model = tf.keras.models.Sequential()
                    # # add dense before
                    # model.add(tf.keras.layers.LSTM(units, input_shape=(units, 1)))
                    # # add dense after
                    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
