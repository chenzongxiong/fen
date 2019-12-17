mu=0
sigma=0
nb_play=1
units=1
__units__=256
points=1000
activation=tanh
method=sin
input_dim=1
state=0
fname="./log/lstm-activation-${activation}-mu-${mu}-sigma-${sigma}-nb_play-${nb_play}-units-${units}-__units__-${__units__}-points-${points}.log"
log_fname="./new-dataset/lstm/method-${method}/activation-${activation}/state-${state}/input_dim-${input_dim}/mu-${mu}/sigma-${sigma}/units-${units}/nb_plays-${nb_plays}/points-${points}/units#-${__units__}/mse-loss.csv"
cat $fname | grep "loss:" | awk '{print $9}' &> log_fname
