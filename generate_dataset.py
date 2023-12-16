import numpy as np

import sys
from pathlib import Path
import os

from mpi4py import MPI
import h5py

import time

import warnings
warnings.simplefilter("ignore")

import spikeinterface.full as si

import preprocessing
import util 


def process_units(recording, peaks_matched, folder, start_unit, end_unit, batch_size=64):
    """
    Processes a dataset in parallel using MPI by iterating over units and saving
    batches of samples and their identifiers in HDF5 file format.
    
    Args:
        recording: A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks_matched (obj): An array containing peaks information with unit ids.
        folder (str): Path to the folder where HDF5 files will be saved.
        batch_size (int, optional): Number of samples to process in each batch. Defaults to 64.
        verbose_interval (int, optional): Interval (in number of batches) at which to print detailed processing progress.
                                          If set to 0 or None, no detailed progress will be printed. Defaults to 10.
    """    
    batch_time_meter = util.AverageMeter()
    
    # MPI initialization
    comm = MPI.COMM_WORLD

    # Get unique unit indices and distribute them among processes
    unit_inds = np.unique(peaks_matched['unit_index'])

    # Iterate over assigned units for each MPI process
    for unit in unit_inds[start_unit:end_unit]:
        hdf5_filename = os.path.join(folder, f"unit_{unit:03}.h5")
        with h5py.File(hdf5_filename, 'w', driver='mpio', comm=comm) as hdf5_file:
            # Filter peaks data for the current unit
            peaks_filtered = peaks_matched[peaks_matched['unit_index'] == unit]
            
            sample_inds = peaks_filtered['sample_index']
            
            total_samples = len(sample_inds)
            total_batches = total_samples // batch_size

            # Pre-allocate datasets for traces and times
            data_shape = (total_samples,) + preprocessing.get_trace_reshaped(recording, sample_inds[0]).shape
            traces_dataset = hdf5_file.create_dataset("traces", shape=data_shape, dtype=np.float32)
            times_dataset = hdf5_file.create_dataset("times", (total_samples,), dtype=np.int32)
            
            min_interval = 1
            verbose_interval = max(min_interval, total_batches // 10) 

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
        print(util.write_separator())


# Main script
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
    
    start_unit = int(sys.argv[3])
    end_unit = int(sys.argv[4])
    
    # Run the dataset generation function
    process_units(recording_preprocessed, data, dataset_folder, start_unit, end_unit)
