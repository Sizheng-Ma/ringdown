#!/bin/sh
#SBATCH -J overtone
#SBATCH -o time_rest/7out_overtone
#SBATCH -e time_rest/error_overtone.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
##SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 time_filter_likelihood_overtone.py
