# SETTINGS
import os

DEBUG_INIT_TF_VALUE = True
WEIGHTS = [1]
WIDTHS = [1]
METHODS = ["sin"]
UNITS = [4, 8, 16]
NB_PLAYS = [1, 2, 3, 4, 8]
EPOCHS = 1000
POINTS = 1000
# NOTE: trick here, batch_size must be always equal to 1
BATCH_SIZE = 1

BATCH_SIZE_LIST = [4, 16, 64, 320, 1600]

FNAME_FORMAT = dict(
    operators="./training-data/operators/{method}-{weight}-{width}.csv",
    operators_predictions="./training-data/operators/{method}-{weight}-{width}-predictions.csv",
    operators_loss="./training-data/operators/{method}-{weight}-{width}-loss.csv",
    operators_gif="./pics/operators/{method}-{weight}-{width}.gif",
    operators_gif_snake="./pics/operators/{method}-{weight}-{width}-snake.gif",

    plays="./training-data/plays/{method}-{weight}-{width}-tanh.csv",
    plays_predictions="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-predictions.csv",
    plays_loss = "./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-loss.csv",
    plays_gif="./pics/plays/{method}-{weight}-{width}-{activation}-{units}.gif",
    plays_gif_snake="./pics/plays/{method}-{weight}-{width}-{activation}-{units}-snake.gif",

    models="./training-data/models/{method}-{weight}-{width}-{nb_plays}.csv",
    models_multi="./training-data/models/{method}-{weight}-{width}-{nb_plays}-multi.csv",
    models_loss="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{batch_size}-loss.csv",
    models_predictions="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-predictions.csv",
    models_multi_predictions="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-multi-predictions.csv",
    models_gif="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}.gif",
    models_multi_gif="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-multi.gif",
    models_gif_snake="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-snake.gif",
    models_multi_gif_snake="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-multi-snake.gif",
)

CPU_COUNTS = min(os.cpu_count(), 32)
