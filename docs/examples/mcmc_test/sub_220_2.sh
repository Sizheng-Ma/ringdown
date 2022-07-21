#!/bin/sh
#SBATCH -J 1mcmc220_2
#SBATCH -o 220_2/out
#SBATCH -e 220_2/error.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 2:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
##SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 220_2_qnm.py
