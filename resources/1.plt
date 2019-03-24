reset
set terminal png
set xlabel "iterations"
set ylabel "error rate"

set title "different number of context"
set key box
set grid

set style data linespoints

set output './1.png'

# set label "17.3%" at 50, 0.172888138561 center point pointtype 1
# set label "27.2%" at 50, 0.272009629213 center point pointtype 2
# set label "71.6%" at 50, 0.716313387107 center point pointtype 3

plot "/Users/baymax_testios/feng/new-dataset/models/diff_weights/method-sin/activation-None/state-0/mu-0/sigma-2/units-20/nb_plays-20/points-1000/input_dim-1/base.csv"  using 1:2 title "no context"
