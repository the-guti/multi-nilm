#!/bin/bash
#
#SBATCH --job-name=nilm-test		# Job name
#SBATCH --output=output.%A_%a.out	# Standard output log
#SBATCH --error=error.%A_%a.err         # Error log config
#SBATCH --nodes=1                  	# Run all processes on a single node	
#SBATCH --ntasks=1			# Run on a single CPU
#SBATCH --mem-per-cpu=4000		# Job memory request
#SBATCH --gres=gpu:1			# Number of GPUs (per node)
#SBATCH -q gpu-single                   # Run it in a single GPU
#SBATCH -p gpu                          # Use GPU

export PYTHONPATH=$(pwd)

cd /home/nicolas.avila/repos/multi-nilm/experiments/

srun python run_my_experiment.py

