#!/bin/sh
#SBATCH -J 7
#SBATCH -o rest/7out
#SBATCH -e rest/error.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 1:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
##SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 run.py
