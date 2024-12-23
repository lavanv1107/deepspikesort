#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J create_dataset
#SBATCH --mail-user=vatanaklavan@proton.me
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00
#SBATCH -o /pscratch/sd/v/vlavan/deep_spike_sort/create_dataset.out
#SBATCH -e /pscratch/sd/v/vlavan/deep_spike_sort/create_dataset.err 
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4

# Change to the main directory
cd /pscratch/sd/v/vlavan/deep_spike_sort

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/v/vlavan/deep_spike_sort

srun -u python -m data.create_dataset sub-CSHL049 peaks