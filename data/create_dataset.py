import argparse
import datetime
import os
import sys
import warnings

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
    recording_folder = f"data/{args.recording_id}/extractors/preprocessed"
    recording_preprocessed = si.load_extractor(recording_folder)
    
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

    # Determine if running within SLURM
    process_id = int(os.getenv('SLURM_PROCID', 0))
    num_tasks = int(os.getenv('SLURM_NTASKS', 1))
    
    # Get unique units and the size of data per unit
    unit_inds, counts = np.unique(data['unit_index'], return_counts=True)
    unit_sizes = dict(zip(unit_inds, counts))
    
    # Sort units by the amount of data, largest first (helps with balancing)
    units_sorted = sorted(unit_sizes.items(), key=lambda x: -x[1])

    # Initialize workload trackers
    workloads = {i: 0 for i in range(num_tasks)}
    task_units = {i: [] for i in range(num_tasks)}

    # Greedily assign units to the least loaded task
    for unit, size in units_sorted:
        # Find the task with the minimum workload
        min_task = min(workloads, key=workloads.get)
        task_units[min_task].append(unit)
        workloads[min_task] += size
        
    units_assigned = task_units[process_id]
    
    # Filter data for assigned units
    data_assigned = data[np.isin(data['unit_index'], units_assigned)]
    
    # Process the units and save data in HDF5 format
    create_dataset(recording_preprocessed, units_assigned, data_assigned, dataset_folder)
    
    
def create_dataset(recording, unit_inds, data, folder, batch_size=128, verbose_count=25):
    """
    Creates a dataset in parallel by iterating over units and saving
    batches of samples and their identifiers in HDF5 file format.
    
    Args:
        recording: A RecordingExtractor object created from an NWB file using SpikeInterface.
        data (obj): An array containing either spikes or peaks information with unit ids.
        folder (str): Path to the folder where HDF5 files will be saved.
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 64.
    """
    # Iterate over each unit assigned to this process
    for unit_idx in unit_inds:
        # Define the filename for the HDF5 file to save data for this unit
        file = os.path.join(folder, f"unit_{unit_idx:03}.h5")
        with h5py.File(file, 'w') as handle:        
            properties = data[list(data.dtype.names)][data['unit_index'] == unit]        

            # Pre-allocate datasets for traces and times in the HDF5 file
            handle.create_dataset('properties', data=properties)
            handle.create_dataset('traces', shape=(len(properties), 64, 192, 2), dtype='<f8')

            num_batches = len(properties) // batch_size
            padding = len(str(num_batches))
            verbose_interval = max(1, num_batches // verbose_count) 

            # Iterate over the samples in batches
            for i in range(0, len(properties), batch_size):
                # Determine the end index for the current batch
                end_idx = min(i + batch_size, len(properties))

                batch_times = properties['time'][i:end_idx]

                # Process data for the current batch
                batch_traces = np.array([get_trace_reshaped(recording, time) for time in batch_times])
                handle['traces'][i:end_idx] = batch_traces

                datetime_formatted = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

                # Log progress every few batches
                batch = i // batch_size
                batch_formatted = f"{batch+1:0{padding}d}"
                if batch % verbose_interval == 0:
                    print(f'{datetime_formatted} - [{unit}][{batch_formatted}/{num_batches}]')   


if __name__ == "__main__":
    args = parse_args()
    main(args)
