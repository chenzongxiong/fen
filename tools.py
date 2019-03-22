import utils
import constants
import log as logging

LOG = logging.getLogger(__name__)



nb_plays = 20
# weights_file_key = 'models_diff_weights_saved_weights'
weights_file_key = 'models_diff_weights_mc_saved_weights'
method = 'sin'
loss_name = 'mse'

mu = 0
sigma = 2
points = 1000
input_dim = 1
# ground truth
nb_plays = 20
units = 1
state = 0
activation = None
# activation = 'tanh'
# predicitons
__nb_plays__ = 20
__nb_plays__ = 20
__units__ = 1
__state__ = 0
__activation__ = None
# __activation__ = 'tanh'



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


def show_weights(weights_fname):
    for i in range(nb_plays):
        LOG.debug("==================== PLAY {} ====================".format(i+1))
        fname = weights_fname[:-3] + '/{}plays/play-{}.h5'.format(nb_plays, i+1)
        utils.read_saved_weights(fname)

def show_loss():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    z = np.linspace(0, 1, 100)
    x = z * np.sin(20 * z)
    y = z * np.cos(20 * z)

    c = x + y

    ax.scatter(x, y, z, c=c)

    plt.show()


if __name__ == "__main__":
    show_loss()
