#!/bin/bash
#SBATCH -N 5
#SBATCH -G 20
#SBATCH -q regular
#SBATCH -J run_dss
#SBATCH --mail-user=vatanaklavan@proton.me
#SBATCH --mail-type=ALL
#SBATCH -t 5:00:00
#SBATCH -A m3513
#SBATCH -o /pscratch/sd/v/vlavan/deep_spike_sort/phase_2/run_dss.out
#SBATCH -e /pscratch/sd/v/vlavan/deep_spike_sort/phase_2/run_dss.err 
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -C gpu&hbm80g

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Change to the main directory
cd /pscratch/sd/v/vlavan/deep_spike_sort

# Set PYTHONPATH to include the main folder
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/v/vlavan/deep_spike_sort

# Run Python script
srun -u python -m phase_2.run_dss sub-CSHL049 1 max 420 0 all 0 mask all_units 0