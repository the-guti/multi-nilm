#!/bin/bash
#
#SBATCH --job-name=1minuk15	# Job name
#SBATCH --output=gpu_output.%A_%a.out	# Standard output log
#SBATCH --error=gpu_error.%A_%a.err         # Error log config
#SBATCH --nodes=1                  	# Run all processes on a single node	
#SBATCH --ntasks=1			# Run on a single CPU
#SBATCH --mem-per-cpu=8000		# Job memory request
#SBATCH -q gpu-8                   # Run it in 4 GPU
#SBATCH -p gpu                          # Use GPU


cd /home/roberto.guillen/Documents/multi-nilm

export PYTHONPATH=$(pwd)

cd /home/roberto.guillen/Documents/multi-nilm/experiments

srun python run_my_experiment.py

