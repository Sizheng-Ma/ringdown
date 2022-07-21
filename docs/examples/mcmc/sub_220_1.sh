#!/bin/sh
#SBATCH -J mcmc220_1
#SBATCH -o 220_1/out
#SBATCH -e 220_1/error.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 2:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
##SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 220_1_qnm.py
