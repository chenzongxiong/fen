import sys
import argparse
import tensorflow as tf

import utils
from play import Play, PlayCell


sess = tf.Session()


def fit(inputs, outputs, units, activation, width, true_weight, method="sin", nbr_of_chunks=1):

    state = tf.random_uniform(shape=(), minval=0, maxval=10, dtype=tf.float32)

    model = tf.keras.Sequential()

    cell = PlayCell(weight=weight, width=width, kernel_constraint=None, debug=False)

    layer = Play(units=units, cell=cell,
                 activation="tanh",
                 nbr_of_chunks=nbr_of_chunks,
                 debug=False, input_shape=(inputs.shape[0],))

    import ipdb; ipdb.set_trace()
    model.add(layer)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    import ipdb; ipdb.set_trace()
    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=["mse"])
    import ipdb; ipdb.set_trace()
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)

    model.fit(inputs, outputs, epochs=500,
              verbose=0,
              batch_size=1200)
              #callbacks=[early_stop])
    loss, mse = model.evaluate(inputs, outputs, verbose=0, batch_size=1200)
    print("\nTesting set Mean Squared Error: {}".format(mse))

    # state = tf.constant(0, dtype=tf.float32)
    # predictions = layer(inputs, state)

    # return sess.run(predictions)
    import ipdb; ipdb.set_trace()
    return None


if __name__ == "__main__":
    utils.writer = utils.get_tf_summary_writer("./log/players/")

    methods = ["sin"]
    widths = [5]
    weights = [2]
    units = 4
    nbr_of_chunks = 1
    activation = "tanh"

    parser = argparse.ArgumentParser()

    parser.add_argument("--generate", dest="generate",
                        required=False,
                        action="store_true")
    parser.add_argument("--train", dest="train",
                        required=False,
                        action="store_true")
    parser.add_argument("--plot", dest="plot",
                        required=False,
                        action="store_true")


    argv = parser.parse_args(sys.argv[1:])

    # if argv.generate:
    #     for method in methods:
    #         for weight in weights:
    #             for width in widths:
    #                 print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    #                 fname = "./training-data/operators/{}-{}-{}.csv".format(method, weight, width)
    #                 inputs, outputs_ = utils.load(fname)
    #                 inputs, outputs = generator(inputs, weight, width, units, activation, nbr_of_chunks)
    #                 if activation is None:
    #                     fname = "./training-data/players/{}-{}-{}-{}-{}-linear.csv".format(method, weight, width, units, nbr_of_chunks)
    #                 else:
    #                     fname = "./training-data/players/{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, nbr_of_chunks, activation)
    #                 utils.save(inputs, outputs, fname)


    # # activation = "tanh"
    # # for method in methods:
    # #     for weight in weights:
    # #         for width in widths:
    # #             print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
    # #             fname = "./training-data/players/{}-{}-{}-{}-{}.csv".format(method, weight, width, units, activation)
    # #             inputs, outputs = utils.load(fname)
    # #             fname = "./pics/players/{}-{}-{}-{}-{}.pdf".format(method, weight, width, units, activation)
    # #             plt.scatter(inputs, outputs)
    # #             plt.savefig(fname)
    # #             # utils.save_animation(inputs, outputs, fname)



    if argv.train:
        activation = "tanh"
        _units = 4
        _nbr_of_chunks = 1

        for method in methods:
            for weight in weights:
                for width in widths:
                    print("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                    fname = "./training-data/players/{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, nbr_of_chunks, activation)
                    inputs, outputs_ = utils.load(fname)
                    # increase *units* in order to increase the capacity of the model
                    predictions = fit(inputs, outputs_, _units, activation, width, weight, method=method,
                                      nbr_of_chunks=_nbr_of_chunks)
                    fname = "./training-data/players/predicted-{}-{}-{}-{}-{}-{}.csv".format(method, weight, width, units, _nbr_of_chunks, activation)
                    utils.save(inputs, predictions, fname)
                    print("========================================")
