#!/bin/sh
#SBATCH -J 1mcmcall
#SBATCH -o all/out
#SBATCH -e all/error.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 2:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
##SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 all.py
