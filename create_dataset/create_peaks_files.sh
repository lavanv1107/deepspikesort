#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J create_dataset
#SBATCH --mail-user=rly@lbl.gov  # Customize the email address to send notifications
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
#SBATCH -o /pscratch/sd/r/rly/deepspikesort/out/create_dataset.out
#SBATCH -e /pscratch/sd/r/rly/deepspikesort/out/create_dataset.err 
#SBATCH -n 16  # <-- Customize the number of tasks to run
#SBATCH -c 16

# Change to the main directory
cd /pscratch/sd/r/rly/deepspikesort

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/r/rly/deepspikesort

module load conda
conda activate /global/common/software/m3513/deepspikesort

# For example, if your data is in `[project root]/data/sub-CSHL049/`, then change the below command to:
# srun -u python -m create_dataset.create_peaks_files sub-CSHL049
srun -u python -m create_dataset.create_peaks_files sub-CSHL049
