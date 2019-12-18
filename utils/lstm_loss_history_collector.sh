diff_weights=1
for nb_plays in 1 50 100 500
do
    units=$nb_plays
    if [[ $nb_plays == 500 ]]; then
        units=100
    fi
    for __units__ in 1 8 16 32 64 128 256
    do
        if [[ $diff_weights == 1 ]]; then
            python utils/lstm_filter_loss.py --units ${units} --nb_plays ${nb_plays} --points 1000 --sigma 0 --mu 0 --lr 0.001 --__units__ ${__units__} --activation tanh --diff-weights &
        else
            python utils/lstm_filter_loss.py --units ${units} --nb_plays ${nb_plays} --points 1000 --sigma 0 --mu 0 --lr 0.001 --__units__ ${__units__} --activation tanh &
        fi
    done
done
