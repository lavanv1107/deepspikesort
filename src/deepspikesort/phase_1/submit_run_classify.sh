#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -q regular
#SBATCH -J run_classify
#SBATCH --mail-user=vatanaklavan@proton.me
#SBATCH --mail-type=ALL
#SBATCH -t 10:00:00
#SBATCH -A m3513
#SBATCH -o /pscratch/sd/v/vlavan/deep_spike_sort/phase_1/run_classify.out
#SBATCH -e /pscratch/sd/v/vlavan/deep_spike_sort/phase_1/run_classify.err 
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Change to the main directory
cd /pscratch/sd/v/vlavan/deep_spike_sort

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/v/vlavan/deep_spike_sort

srun -u python -m phase_1.run_classify 