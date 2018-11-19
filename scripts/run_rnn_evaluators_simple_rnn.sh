#!/bin/bash

#SBATCH -J rnn_evaluators_simple_rnn
#SBATCH -D /home/zxchen/feng
#SBATCH -o ./tmp/rnn_evaluators_simple_rnn.%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=30-00:00:00
#SBATCH --partition=big

hostname
source /home/zxchen/.venv3/bin/activate
python rnn_evaluators.py --neural-type simple_rnn
