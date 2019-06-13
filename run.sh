#!/bin/bash

# sbatch scripts/run_rnn_evaluators_simple_rnn.sh
# sbatch scripts/run_rnn_evaluators_lstm.sh
# sbatch scripts/run_rnn_evaluators_gru.sh

sbatch scripts/run_MLE-0.sh
sbatch scripts/run_MLE-1.sh
sbatch scripts/run_MLE-2.sh
sbatch scripts/run_MLE-3.sh
sbatch scripts/run_MLE-4.sh
sbatch scripts/run_MLE-5.sh
sbatch scripts/run_MLE-6.sh
sbatch scripts/run_MLE-7.sh
sbatch scripts/run_MLE-8.sh
sbatch scripts/run_MLE-9.sh


# python dataset_generator.py --operator
# python dataset_generator.py --play
# python dataset_generator.py --model --nb_plays 1 --units 1 &
# python dataset_generator.py --model --nb_plays 1 --units 8 &
# python dataset_generator.py --model --nb_plays 1 --units 20 &
# python dataset_generator.py --model --nb_plays 1 --units 100 &

# python dataset_generator.py --model --nb_plays 20 --units 1 &
# python dataset_generator.py --model --nb_plays 20 --units 8 &
# python dataset_generator.py --model --nb_plays 20 --units 20 &
# python dataset_generator.py --model --nb_plays 20 --units 100 &
# python dataset_generator.py --model --nb_plays 40 --units 1 &
# python dataset_generator.py --model --nb_plays 40 --units 8 &
# python dataset_generator.py --model --nb_plays 40 --units 20 &
# python dataset_generator.py --model --nb_plays 40 --units 100 &
# python dataset_generator.py --model --nb_plays 100 --units 1 &
# python dataset_generator.py --model --nb_plays 100 --units 8 &
# python dataset_generator.py --model --nb_plays 100 --units 20 &
# python dataset_generator.py --model --nb_plays 100 --units 100 &
