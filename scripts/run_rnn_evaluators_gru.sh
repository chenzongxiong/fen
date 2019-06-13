#!/bin/bash

#SBATCH -J rnn_evaluators_gru
#SBATCH -D /home/zxchen/feng
#SBATCH -o ./tmp/rnn_evaluators_gru.out
#SBATCH --cpus-per-task=16
#SBATCH --time=30-00:00:00
#SBATCH --partition=big
#SBATCH --mem=100G

hostname
source /home/zxchen/.venv3/bin/activate
python rnn_evaluators.py --neural-type gru
