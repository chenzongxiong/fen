#!/bin/bash

sbatch scripts/run_rnn_evaluators_simple_rnn.sh
sbatch scripts/run_rnn_evaluators_lstm.sh
sbatch scripts/run_rnn_evaluators_gru.sh
