# SETTINGS
import os

DEBUG_INIT_TF_VALUE = False
WEIGHTS = [1]
WIDTHS = [1]
METHODS = ["sin"]
# UNITS = [4, 8, 16]
UNITS = [1, 8, 20, 100]
# NB_PLAYS = [1, 4, 10, 20]
NB_PLAYS = [1, 20 , 40, 100]

EPOCHS = 20000
POINTS = 500
# NOTE: trick here, batch_size must be always equal to 1
BATCH_SIZE = 1

BATCH_SIZE_LIST = [10]

FNAME_FORMAT = dict(
    operators="./training-data/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/base.csv",
    operators_predictions="./training-data/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/predictions-{loss}.csv",
    operators_loss="./training-data/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/loss-{loss}.csv",
    operators_gif="./pics/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/base-{loss}.gif",
    operators_gif_snake="./pics/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/snake-{loss}.gif",

    plays="./training-data/plays/method-{method}/weight-{weight}/width-{width}/points-{points}/base.csv",
    plays_predictions="./training-data/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/predictions-{loss}.csv",
    plays_loss = "./training-data/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/loss-{loss}.csv",
    plays_gif="./pics/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/base-{loss}.gif",
    plays_gif_snake="./pics/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/snake-{loss}.gif",

    models="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/points-{points}/base.csv",
    models_loss="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/batch_size-{batch_size}/points-{points}/loss-{loss}.csv",
    models_loss_history="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/batch_size-{batch_size}/points-{points}/history-{loss}.csv",
    models_predictions="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/batch_size-{batch_size}/points-{points}/predictions-{loss}.csv",
    models_gif="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/points-{points}/base-{loss}.gif",
    models_gif_snake="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/points-{points}/snake-{loss}.gif",

    models_rnn_loss="./training-data/models/rnn/{method}-{weight}-{width}-{nb_plays}-rnn-loss.csv",
    models_rnn_predictions="./training-data/models/rnn/{method}-{weight}-{width}-{nb_plays}-rnn-predictions.csv",
    models_lstm_loss="./training-data/models/lstm/{method}-{weight}-{width}-{nb_plays}-lstm-loss.csv",
    models_lstm_predictions="./training-data/models/lstm/{method}-{weight}-{width}-{nb_plays}-lstm-predictions.csv",
    models_gru_loss="./training-data/models/gru/{method}-{weight}-{width}-{nb_plays}-gru-loss.csv",
    models_gru_predictions="./training-data/models/gru/{method}-{weight}-{width}-{nb_plays}-gru-predictions.csv",

    # G model
    models_G="./training-data/G/{method}-{weight}-{width}-{nb_plays}.csv",
    # models_G_multi="./training-data/G/{method}-{weight}-{width}-{nb_plays}-multi.csv",
    models_G_loss="./training-data/G/{method}-{weight}-{width}-{nb_plays}-{batch_size}-loss.csv",
    models_G_predictions="./training-data/G/{method}-{weight}-{width}-{nb_plays}-{batch_size}-predictions.csv",
    # F model
    models_F="./training-data/F/{method}-{weight}-{width}-{nb_plays}.csv",
    # models_F_multi="./training-data/F/{method}-{weight}-{width}-{nb_plays}-multi.csv",
    models_F_loss="./training-data/F/{method}-{weight}-{width}-{nb_plays}-{batch_size}-loss.csv",
    models_F_predictions="./training-data/F/{method}-{weight}-{width}-{nb_plays}-{batch_size}-predictions.csv",
    models_F_gif="./pics/F/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}.gif",
    models_F_gif_snake="./pics/F/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-snake.gif",


    # operator noise
    operators_noise="./training-data/operators/{method}-{weight}-{width}-{mu}-{sigma}.csv",
    operators_noise_predictions="./training-data/operators/{method}-{weight}-{width}-{mu}-{sigma}-predictions.csv",
    operators_noise_loss="./training-data/operators/{method}-{weight}-{width}-{mu}-{sigma}-loss.csv",
    operators_noise_loss_histroy="./training-data/operators/{method}-{weight}-{width}-{mu}-{sigma}-loss-history.csv",
    operators_noise_gif="./pics/operators/{method}-{weight}-{width}-{mu}-{sigma}.gif",
    operators_noise_gif_snake="./pics/operators/{method}-{weight}-{width}-{mu}-{sigma}-snake.gif",

    # play noise
    plays_noise="./training-data/plays/{method}-{weight}-{width}-{mu}-{sigma}-tanh.csv",
    plays_noise_predictions="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-predictions.csv",
    plays_noise_loss="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-loss.csv",
    plays_noise_loss_history="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-loss-history.csv",
    plays_noise_gif="./pics/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}.gif",
    plays_noise_gif_snake="./pics/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-snake.gif",

    # model noise
    models_noise="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{mu}-{sigma}.csv",
    models_noise_loss_history="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{batch_size}-{mu}-{sigma}-loss-history.csv",
    models_noise_loss="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{batch_size}-{mu}-{sigma}-loss.csv",
    models_noise_predictions="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-{mu}-{sigma}-predictions.csv",
    # models_noise_multi_predictions="./training-data/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-multi-predictions.csv",
    models_noise_gif="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-{mu}-{sigma}.gif",
    # models_noise_multi_gif="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-multi.gif",
    models_noise_gif_snake="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-{mu}-{sigma}-snake.gif",
    # models_noise_multi_gif_snake="./pics/models/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-multi-snake.gif",

    mc="./training-data/mc/points-{points}/mu-{mu}/sigma-{sigma}/base.csv",
    F="./training-data/F/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.csv",
    F_loss_history="./training-data/F/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-loss-history-{loss}.csv",
    F_predictions="./training-data/F/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/predictions-{loss}.csv",
    F_gif="./pics/F/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.gif",
    F_gif_snake="./pics/F/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/snake-{loss}.gif",
    G_predictions="./training-data/G/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/predictions-{loss}.csv",
    G_loss_history="./training-data/G/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-loss-history-{loss}.csv",
    G_gif="./pics/G/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.gif",
    G_gif_snake="./pics/F/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/snake-{loss}.gif",
)

CPU_COUNTS = min(os.cpu_count(), 32)


class NetworkType:
    OPERATOR = 1
    PLAY = 2


LOG_DIR = "./log"
