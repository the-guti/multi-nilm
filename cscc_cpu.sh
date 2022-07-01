#!/bin/bash
#
#SBATCH --job-name=ukparallel		# Job name
#SBATCH --output=cpu_output.%A_%a.out	# Standard output log
#SBATCH --error=cpu_error.%A_%a.err         # Error log config
#SBATCH --nodes=1                  	# Run all processes on a single node	
#SBATCH --ntasks=1			# Run on a single CPU
#SBATCH --mem-per-cpu=8000		# Job memory request
#SBATCH -q cpu-512                       
#SBATCH -p cpu                         

cd /home/roberto.guillen/Documents/multi-nilm

export PYTHONPATH=$(pwd)
conda activate nilmtk-env
srun python experiments/test.py
# srun python experiments/run_multiple_exp.py
# srun python experiments/run_my_experiment.py
