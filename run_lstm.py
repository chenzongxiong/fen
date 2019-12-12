import sys
import argparse


import trading_data as tdata
import tensorflow as tf
import numpy as np


# units = [1500]
# capcity  = [1, 2, 4, 8, 16, 32]
def lstm_return_sequence(input_fname, units, capacity=1, epochs=5000):
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)
    _timestamp = np.arange(2000)
    units = 1500

    prices = _prices[:units]
    timestamp = _timestamp[:units].reshape(1, -1, 1)
    prices = prices.reshape(1, -1, 1)   # (batch_size, timestamp, input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(int(units*capacity),
                                   input_shape=(units, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    model.fit(timestamp, prices, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback])

    pred_inputs = np.hstack([_timestamp[units:2000], _timestamp[:1000]])
    truth_outputs = np.hstack([_prices[units:2000], _prices[:1000]])

    # pred_inputs = _timestamp[:units]
    # truth_outputs = _prices[:units].reshape(-1)

    pred_inputs = pred_inputs.reshape(1, -1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs[:500] - pred_outputs[:500]) ** 2)

    print("LSTM mse: {}".format(mse))

    output_fname = "new-dataset/lstm/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=capacity)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)


def lstm_without_sequence(inputs_fname, units, epochs=5000):
    units = 1500
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)
    _timestamp = np.arange(2000)
    prices = _prices[:units]
    timestamp = _timestamp[:units]
    timestamp = timestamp[:units].reshape(1, -1, 1)
    prices = prices.reshape(1, -1)   # (batch_size, timestamp, input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(int(units),
                                   input_shape=(units, 1),
                                   unroll=False,
                                   return_sequences=False,
                                   use_bias=True,
                                   implementation=2))
    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    model.fit(timestamp, prices, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback])

    pred_inputs = np.hstack([_timestamp[units:2000], _timestamp[:1000]])
    truth_outputs = np.hstack([_prices[units:2000], _prices[:1000]])

    # pred_inputs = _timestamp[:units]
    # truth_outputs = _prices[:units]

    pred_inputs = pred_inputs.reshape(1, -1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs - pred_outputs) ** 2)
    print("LSTM mse: {}".format(mse))

    output_fname = "new-dataset/lstm/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=0)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)



if __name__ == "__main__":

    input_fname = "new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv"


    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", dest="capacity",
                        required=False, default=1,
                        type=int)

    parser.add_argument("--epochs", dest="epochs",
                        required=False, default=100,
                        type=int)

    parser.add_argument("--method", dest="method",
                        required=False, default=1,
                        type=int)

    argv = parser.parse_args(sys.argv[1:])
    method = argv.method
    units = 1500
    epochs = argv.epochs
    capacity = argv.capacity

    if method == 1:
        lstm_return_sequence(input_fname, units, capacity=capacity, epochs=epochs)
    elif method == 2:
        lstm_without_sequence(input_fname, units, epochs=epochs)
    else:
        pass
