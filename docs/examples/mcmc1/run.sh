#!/bin/bash
str1='1.7'
str2='0.84'
sed -i "s/tinit=${str1}/tinit=${str2}/" all.py
sed -i "s/tinit=${str1}/tinit=${str2}/" 221_1_qnm.py
sed -i "s/tinit=${str1}/tinit=${str2}/" 220_2_qnm.py
sed -i "s/tinit=${str1}/tinit=${str2}/" 2201_qnm.py
sed -i "s/tinit=${str1}/tinit=${str2}/" 220_1_qnm.py

sbatch sub_220_2.sh
sbatch sub_220_1.sh
sbatch sub_2201.sh
sbatch sub_221_1.sh
sbatch sub_all.sh
