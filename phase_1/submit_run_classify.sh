#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -q regular
#SBATCH -J run_classify
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00
#SBATCH -A 
#SBATCH -o /path/to/run_classify.out
#SBATCH -e /path/to/run_classify.err 
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/path/to/main_folder

srun -u python -m phase_1.run_classify 