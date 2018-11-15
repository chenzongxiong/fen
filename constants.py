# SETTINGS
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
    plays="./training-data/plays/{method}-{weight}-{width}-tanh.csv",
    plays_predictions="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-predictions.csv",

    models="./training-data/models/{method}-{weight}-{width}-{nb_plays}.csv",
    models_multi="./training-data/models/{method}-{weight}-{width}-{nb_plays}-multi.csv",
    models_loss="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{batch_size}loss.csv",
    models_predictions="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{batch_size}-predictions.csv",
    models_multi_predictions="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{batch_size}-multi-predictions.csv",
)
