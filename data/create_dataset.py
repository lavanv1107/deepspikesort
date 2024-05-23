import argparse
import os
import sys
import time
import warnings

from collections import Counter

import numpy as np
import h5py

import spikeinterface.full as si

warnings.simplefilter("ignore")

sys.path.append("..")
from preprocessing import get_trace_reshaped
from util import AverageMeter, print_separator


def parse_args():
    parser = argparse.ArgumentParser(description="Process units and save data in HDF5 format.")
    
    # Retrieve parameters from command line arguments
    parser.add_argument('recording_id', type=str, help="ID of the recording.")
    parser.add_argument('dataset_type', type=str, choices=['spikes', 'peaks', 'noise'], help="Type of dataset: 'spikes', 'peaks', or 'noise'.")
    parser.add_argument('num_processes', type=int, help="Number of processes to distribute units across.")
    parser.add_argument('process_id', type=int, help="Process ID for distributing units.")
    
    return parser.parse_args()


def main(args):    
    # Load the preprocessed recording using SpikeInterface
    preprocessed_folder = f"data/{args.recording_id}/extractors/preprocessed"
    recording_preprocessed = si.load_extractor(preprocessed_folder)
    
    # Determine the dataset folder and file based on the dataset type
    if args.dataset_type == 'spikes':
        dataset_folder = f'data/{args.recording_id}/spikes'
        data_file = os.path.join(dataset_folder, "spikes.npy")  
    elif args.dataset_type == 'noise':
        dataset_folder = f'data/{args.recording_id}/spikes'
        data_file = os.path.join(dataset_folder, "noise.npy")
    elif args.dataset_type == 'peaks':
        dataset_folder = f'data/{args.recording_id}/peaks'
        data_file = os.path.join(dataset_folder, "peaks_matched.npy")
        
    # Load data from the file
    data = np.load(data_file)    

    # Distribute the units among the processes
    dist_units = distribute_units(data['unit_index'], args.num_processes)
    
    # Get the assigned units for the current process ID
    assigned_units = dist_units[args.process_id]
    
    # Process the units and save data in HDF5 format
    create_dataset(recording_preprocessed, data, dataset_folder, assigned_units)
    
    
def create_dataset(recording, data, folder, process, batch_size=128):
    """
    Creates a dataset in parallel by iterating over units and saving
    batches of samples and their identifiers in HDF5 file format.
    
    Args:
        recording: A RecordingExtractor object created from an NWB file using SpikeInterface.
        data (obj): An array containing either spikes or peaks information with unit ids.
        folder (str): Path to the folder where HDF5 files will be saved.
        process (dict): Dictionary containing 'unit_inds' (list of unit indices assigned to this process)
                        and 'total_samples' (total number of samples assigned to this process).
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 64.
    """
    batch_time_meter = AverageMeter()
    
    # Get the list of unit indices and total number of samples assigned to this process
    unit_inds = process['unit_inds']
    total_samples = process['total_samples']
    
    print(f'Units in process: {unit_inds}')
    print(f'Samples in process: {total_samples}')
    
    # Iterate over each unit assigned to this process
    for unit in unit_inds:
        print(write_separator())
        print(f'Unit {unit}')
        print(write_separator())
        
        # Define the filename for the HDF5 file to save data for this unit
        hdf5_filename = os.path.join(folder, f"unit_{unit:03}.h5")
        
        # Open the HDF5 file in write mode
        with h5py.File(hdf5_filename, 'w') as hdf5_file:
            # Filter the data for the current unit
            data_filtered = data[data['unit_index'] == unit]
            
            # Get the sample indices for the current unit
            sample_inds = data_filtered['sample_index']
            
            # Calculate the total number of samples and batches
            total_samples = len(sample_inds)
            total_batches = total_samples // batch_size

            # Pre-allocate datasets for traces and times in the HDF5 file
            data_shape = (total_samples,) + get_trace_reshaped(recording, sample_inds[0]).shape
            traces_dataset = hdf5_file.create_dataset("traces", shape=data_shape, dtype=np.float32)
            times_dataset = hdf5_file.create_dataset("times", (total_samples,), dtype=np.int32)
            
            # Define intervals for logging progress
            min_interval = 1
            verbose_interval = max(min_interval, total_batches // 50) 

            end = time.time()
            # Iterate over the samples in batches
            for i in range(0, total_samples, batch_size):
                # Determine the end index for the current batch
                batch_end = min(i + batch_size, total_samples)
                batch_samples = sample_inds[i:batch_end]

                # Process data for the current batch
                batch_traces = np.array([get_trace_reshaped(recording, sample) for sample in batch_samples])
                traces_dataset[i:batch_end] = batch_traces

                # Assign time identifiers for each sample in the batch
                batch_times = [sample for sample in batch_samples]  
                times_dataset[i:batch_end] = batch_times

                # Update and log time taken for the batch
                batch_time_meter.update(time.time() - end)
                end = time.time()
                
                # Log progress every few batches
                batch = i // batch_size
                if batch % verbose_interval == 0:
                    print(f'Unit: [{unit}][{batch}/{total_batches}]\t'
                          f'Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})')
        
        
def distribute_units(unit_inds, num_processes):
    """
    Distributes units among a specified number of tasks to balance the workload.

    Args:
        unit_inds (list): A list of unit indices to be distributed.
        num_processes (int, optional): The number of processes to distribute the units across.

    Returns:
        dict: A dictionary where each key is a process identifier (from 1 to num_processes),
              and each value is a dictionary containing:
              - 'unit_inds' (list): The list of unit indices assigned to the process.
              - 'total_samples' (int): The total number of samples assigned to the process.
    """
    # Count the occurrences of each unit index to get the number of samples per unit
    unit_counts = dict(Counter(unit_inds))
    
    # Calculate the total number of samples
    total_samples = sum(unit_counts.values())
    
    # Calculate the target number of samples per process for even distribution
    target_samples_per_process = total_samples / num_processes

    # Initialize a dictionary to store the processes and their assigned units and samples
    processes = {i: {'unit_inds': [], 'total_samples': 0} for i in range(1, num_processes + 1)}

    # Sort units by the number of samples in descending order to allocate larger units first
    sorted_units = sorted(unit_counts.items(), key=lambda x: x[1], reverse=True)

    # Distribute units among processes
    for unit, samples in sorted_units:
        # Find the process with the minimum total samples assigned so far
        min_process = min(processes, key=lambda x: processes[x]['total_samples'])
        
        # Assign the current unit to this process
        processes[min_process]['unit_inds'].append(unit)
        
        # Update the total samples for this process
        processes[min_process]['total_samples'] += samples

    return processes


if __name__ == "__main__":
    args = parse_args()
    main(args)
