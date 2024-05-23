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
from util import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Process units and save data in HDF5 format.")
    # Retrieve parameters from command line arguments
    parser.add_argument('recording_id', type=str, help="ID of the recording.")
    parser.add_argument('dataset_type', type=str, choices=['spikes', 'peaks', 'noise'], help="Type of dataset: 'spikes', 'peaks', or 'noise'.")
    
    return parser.parse_args()


def main(args):    
    # Load the preprocessed recording using SpikeInterface
    preprocessed_folder = f"data/{args.recording_id}/extractors/preprocessed"
    recording_preprocessed = si.load_extractor(preprocessed_folder)
    
    # Determine the dataset folder and file based on the dataset type
    if args.dataset_type == 'spikes':
        dataset_folder = f'data/{args.recording_id}/spikes'
        data_file = os.path.join(dataset_folder, "spikes_test.npy")  
    elif args.dataset_type == 'noise':
        dataset_folder = f'data/{args.recording_id}/spikes'
        data_file = os.path.join(dataset_folder, "noise.npy")
    elif args.dataset_type == 'peaks':
        dataset_folder = f'data/{args.recording_id}/peaks'
        data_file = os.path.join(dataset_folder, "peaks_matched.npy")
        
    # Load data from the file
    data = np.load(data_file)    

    # Determine if running within SLURM
    process_id = int(os.getenv('SLURM_PROCID', 0))
    num_tasks = int(os.getenv('SLURM_NTASKS', 1))
    
    # Determine the subset of units for this process
    unit_indices = np.unique(data['unit_index'])
    units_per_task = len(unit_indices) // num_tasks
    start_idx = process_id * units_per_task
    end_idx = start_idx + units_per_task if process_id != num_tasks - 1 else len(unit_indices)
    assigned_units = unit_indices[start_idx:end_idx]
    
    # Filter data for assigned units
    assigned_data = data[np.isin(data['unit_index'], assigned_units)]
    
    # Store process information
    process_info = {'unit_inds': assigned_units, 'total_samples': len(assigned_data)}
    
    # Process the units and save data in HDF5 format
    create_dataset(recording_preprocessed, assigned_data, dataset_folder, process_info)
    
    
def create_dataset(recording, data, folder, process, batch_size=128):
    """
    Creates a dataset in parallel by iterating over units and saving
    batches of samples and their identifiers in HDF5 file format.
    
    Args:
        recording: A RecordingExtractor object created from an NWB file using SpikeInterface.
        data (obj): An array containing either spikes or peaks information with unit ids.
        folder (str): Path to the folder where HDF5 files will be saved.
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 64.
    """
    batch_time_meter = AverageMeter()
    
    # Get the list of unit indices and total number of samples assigned to this process
    unit_inds = process['unit_inds']
    total_samples = process['total_samples']
    
    # Iterate over each unit assigned to this process
    for unit in unit_inds:
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
