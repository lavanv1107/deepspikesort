#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J create_dataset
#SBATCH --mail-user=  # Customize the email address to send notifications
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
#SBATCH -o /path/to/create_dataset.out
#SBATCH -e /path/to/create_dataset.err 
#SBATCH -n 16  # <-- Customize the number of tasks to run
#SBATCH -c 16

# Change to the main directory
cd /path/to/main_folder

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/path/to/main_folder

# For example, if your data is in `[project root]/data/sub-CSHL049/`, then change the below command to:
# srun -u python -m create_dataset.create_peaks_files sub-CSHL049
srun -u python -m create_dataset.create_peaks_files ...