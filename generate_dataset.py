import sys
import os

import numpy as np
import h5py

from mpi4py import MPI

from collections import Counter

import time

import warnings
warnings.simplefilter("ignore")

import spikeinterface.full as si

import preprocessing
import util 


def process_units(recording, data, folder, process, batch_size=64):
    """
    Processes a dataset in parallel using MPI by iterating over units and saving
    batches of samples and their identifiers in HDF5 file format.
    
    Args:
        recording: A RecordingExtractor object created from an NWB file using SpikeInterface.
        data (obj): An array containing either spikes or peaks information with unit ids.
        folder (str): Path to the folder where HDF5 files will be saved.
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 64.
    """        
    comm = MPI.COMM_WORLD
    
    batch_time_meter = util.AverageMeter()
    
    unit_inds = process['unit_inds']
    total_samples = process['total_samples']
    
    print(f'Units in process: {unit_inds}')
    print(f'Samples in process: {total_samples}')
    
    # Iterate over assigned units for each MPI process
    for unit in unit_inds:
        print(util.write_separator())
        print(f'Unit {unit}')
        print(util.write_separator())
        
        hdf5_filename = os.path.join(folder, f"unit_{unit:03}.h5")
        with h5py.File(hdf5_filename, 'w', driver='mpio', comm=comm) as hdf5_file:
            # Filter data for the current unit
            data_filtered = data[data['unit_index'] == unit]
            
            sample_inds = data_filtered['sample_index']
            
            total_samples = len(sample_inds)
            total_batches = total_samples // batch_size

            # Pre-allocate datasets for traces and times
            data_shape = (total_samples,) + preprocessing.get_trace_reshaped(recording, sample_inds[0]).shape
            traces_dataset = hdf5_file.create_dataset("traces", shape=data_shape, dtype=np.float32)
            times_dataset = hdf5_file.create_dataset("times", (total_samples,), dtype=np.int32)
            
            min_interval = 1
            verbose_interval = max(min_interval, total_batches // 50) 

            # Process and save data in batches
            end = time.time()
            for i in range(0, total_samples, batch_size):
                # Determine the end index for the current batch
                batch_end = min(i + batch_size, total_samples)
                batch_samples = sample_inds[i:batch_end]

                # Process data for the current batch
                batch_traces = np.array([preprocessing.get_trace_reshaped(recording, sample) for sample in batch_samples])
                traces_dataset[i:batch_end] = batch_traces

                # Assign time identifiers for each sample in the batch
                batch_times = [sample for sample in batch_samples]  
                times_dataset[i:batch_end] = batch_times

                # Update and log time taken for the batch
                batch_time_meter.update(time.time() - end)
                end = time.time()
                
                batch = i // batch_size
                if batch % verbose_interval == 0:
                    print(f'Unit: [{unit}][{batch}/{total_batches}]\t'
                          f'Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})')
        
        
def distribute_processes(unit_inds, num_processes=32):
    unit_counts = dict(Counter(unit_inds))
    
    # Calculate total size and target size per session
    total_samples = sum(unit_counts.values())
    target_samples_per_process = total_samples / num_processes

    # Initialize containers for processes
    processes = {i: {'unit_inds': [], 'total_samples': 0} for i in range(1, num_processes+1)}

    # Sort units by size in descending order to allocate larger units first
    sorted_units = sorted(unit_counts.items(), key=lambda x: x[1], reverse=True)

    # Distribute units among processes
    for unit, samples in sorted_units:
        # Find the session with the minimum total size at the moment
        min_process = min(processes, key=lambda x: processes[x]['total_samples'])
        # Add unit to this session
        processes[min_process]['unit_inds'].append(unit)
        # Update the total size of this session
        processes[min_process]['total_samples'] += samples

    return processes


def create_noise(recording, spikes):
    samples_range = np.arange(31, recording.get_num_frames() - 33)
    mask = ~np.isin(samples_range, spikes['sample_index'])
    noise_samples = samples_range[mask]
    selected_samples = np.random.choice(noise_samples, size=100000, replace=False)
    
    # Define the structured data type
    dtype = [('unit_index', '<i8'), ('sample_index', '<i8')]

    # Create an empty array with the structured dtype
    noise = np.empty(len(selected_samples), dtype=dtype)
    
    # Populate unit_index with -1
    noise['unit_index'] = -1 

    # Put noise samples in sample_index  
    noise['sample_index'] = selected_samples
    
    return noise


if __name__ == "__main__":
    # Read preprocessed recording object
    recording_num = int(sys.argv[1])
    preprocessed_folder = f"extractors/{recording_num:03}/preprocessed"
    recording_preprocessed = si.load_extractor(preprocessed_folder)

    dataset_type = int(sys.argv[2])
    if dataset_type == 0:
        # Load spikes object
        dataset_folder = f'spikes/{recording_num:03}'
        data_file = os.path.join(dataset_folder, "spikes_nwb.npy")
        data = np.load(data_file)
    elif dataset_type == 1:
        # Load matched peaks object
        dataset_folder = f'peaks/{recording_num:03}'
        data_file = os.path.join(dataset_folder, "peaks_matched.npy")
        data = np.load(data_file)     
    elif dataset_type == 2:
        # Create array of noise samples
        dataset_folder = f'spikes/{recording_num:03}'
        data_file = os.path.join(dataset_folder, "spikes_nwb.npy")
        spikes = np.load(data_file)
        data = create_noise(recording_preprocessed, spikes)
    
    dist_processes = distribute_processes(data['unit_index'])

    # Determine the current process's identifier
    process_id = int(sys.argv[3])

    # Get the assigned process
    assigned_process = dist_processes[process_id]
    
    # Run the dataset generation function
    process_units(recording_preprocessed, data, dataset_folder, assigned_process)
