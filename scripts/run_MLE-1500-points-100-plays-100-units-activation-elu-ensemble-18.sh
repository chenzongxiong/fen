#!/bin/bash

#SBATCH -J run_MLE-nb_plays-100-units-100-activation-elu-batch-size-1500-ensemble-18
#SBATCH -D /home/zxchen/feng
#SBATCH -o ./tmp/run_MLE-nb_plays-100-units-100-activation-elu-batch-size-1500-ensemble-18.out
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=30-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com

hostname
source /home/zxchen/.venv3/bin/activate
python run_MLE.py --__nb_plays__ 100 --__units__ 100 --__activation__ elu --batch_size 1500 --ensemble 18
# 0.0001
