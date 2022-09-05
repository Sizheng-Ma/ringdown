#!/bin/sh
#SBATCH -J mean2
#SBATCH -o time_rest/7out
#SBATCH -e time_rest/error.txt
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 2:00:00
#SBATCH --mail-user=sma@caltech.edu
#SBATCH --mail-type=END
#SBATCH -p debug

source activate ringdown
~/.conda/envs/ringdown/bin/python3 -u mean2.py
