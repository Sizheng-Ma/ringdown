#!/bin/sh
#SBATCH -J mcmc2201
#SBATCH -o 2201/out
#SBATCH -e 2201/error.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 2:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
##SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 2201_qnm.py
