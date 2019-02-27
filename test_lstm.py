import tensorflow as tf
import numpy as np


if __name__ == "__main__":

    size = [1, 1, 64]

    x = np.random.normal(size=size)
    y = np.random.normal(size=size)
    x = x.reshape(1, 64, 1)
    y = y.reshape(1, 64, 1)
    # y = y.reshape(1, -1)
    # units = y.shape[-1]

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)

    model = tf.keras.models.Sequential()
    # units = 32
    units = 1
    # outputs = tf.keras.layers.LSTM(1, input_shape=(64, 1))
    model.add(tf.keras.layers.LSTM(1, input_shape=(64, 1), return_sequences=True))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    model.fit(x, y, verbose=1, epochs=1, batch_size=2)
