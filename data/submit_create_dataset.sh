#!/bin/bash
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J create_dataset
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00
#SBATCH -o /path/to/create_dataset.out
#SBATCH -e /path/to/create_dataset.err 
#SBATCH -n 32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/path/to/main_folder

srun -u python -m data.create_dataset sub-CSHL049 spikes $SLURM_NTASKS ${SLURM_PROCID}
