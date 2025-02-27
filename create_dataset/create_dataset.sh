#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J create_dataset
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
#SBATCH -o /path/to/create_dataset.out
#SBATCH -e /path/to/create_dataset.err 
#SBATCH -n 16
#SBATCH -c 16

# Change to the main directory
cd /path/to/main_folder

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/path/to/main_folder

srun -u python -m data.create_dataset ...