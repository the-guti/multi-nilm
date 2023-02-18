#!/bin/bash
#
#SBATCH --job-name=sql_para	# Job name
#SBATCH --output=cluster_output/test_script.%A_%a.out	# Standard output log
#SBATCH --error=cluster_output/test_script.%A_%a.err         # Error log config
#SBATCH --nodes=1                  	# Run all processes on a single node	
#SBATCH --ntasks=1			# Run on a single CPU
#SBATCH -q cpu-512                       
#SBATCH -p cpu                         
#SBATCH --nodelist=cn-08

cd /home/roberto.guillen/Documents/multi-nilm

export PYTHONPATH=$(pwd)
conda activate nilmtk-env
srun python experiments/run.py
