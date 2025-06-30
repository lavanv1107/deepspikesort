#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J dss_create_cshl049_dataset
#SBATCH --mail-user=rly@lbl.gov  # Customize the email address to send notifications
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
#SBATCH -A m3513  # Set the project ID
#SBATCH -o /pscratch/sd/r/rly/deepspikesort/out/create_dataset.out  # Customize the output log locations
#SBATCH -e /pscratch/sd/r/rly/deepspikesort/out/create_dataset.err 
#SBATCH -n 16  # <-- Customize the number of tasks to run
#SBATCH -c 16

# Change to the main directory
cd /pscratch/sd/r/rly/deepspikesort

# Load a conda (or other) environment where deepspikesort is installed
module load conda
conda activate /global/common/software/m3513/deepspikesort2

# Run create_peaks_files.py
srun -u python src/deepspikesort/create_dataset/create_peaks_files.py sub-CSHL049
