#!/bin/bash
#
#SBATCH --job-name=nilm-test		# Job name
#SBATCH --output=output.%A_%a.out	# Standard output log
#SBATCH --error=error.%A_%a.err         # Error log config
#SBATCH --nodes=1                  	# Run all processes on a single node	
#SBATCH --ntasks=1			# Run on a single CPU
#SBATCH --mem-per-cpu=8000		# Job memory request
#SBATCH -q cpu-512                   # Run it in a single GPU
#SBATCH -p cpu                          # Use GPU
cd /home/roberto.guillen/Documents/multi-nilm

export PYTHONPATH=$(pwd)

conda activate nilmtk-env
srun python experiments/test.py

