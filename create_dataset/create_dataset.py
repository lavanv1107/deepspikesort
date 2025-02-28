import argparse
import os
import sys
import warnings

import numpy as np
import h5py

import spikeinterface.full as si

warnings.simplefilter("ignore")

sys.path.append("..")
from preprocessing import get_trace_reshaped


def parse_args():
    parser = argparse.ArgumentParser(description="Create peaks dataset in HDF5 format.")
    parser.add_argument('recording_id', type=str, help="ID of the recording.")
    return parser.parse_args()


def main(args):    
    # Get this process's ID and the total number of tasks
    process_id = int(os.getenv('SLURM_PROCID', 0))
    num_tasks = int(os.getenv('SLURM_NTASKS', 1))
    
    # Load input data
    data_folder = f'data/{args.recording_id}'
    
    recording = si.load_extractor(os.path.join(data_folder, "extractors/preprocessed"))
    peaks = np.load(os.path.join(data_folder, "peaks/peaks_test.npy"))

    # Calculate chunk for this process
    chunk_size = len(peaks) // num_tasks
    start_idx = process_id * chunk_size
    end_idx = start_idx + chunk_size if process_id < num_tasks - 1 else len(peaks)
    
    # Create output file for this process
    output_file = os.path.join(data_folder, f"peaks/peaks_{process_id}.h5")
    process_peaks(recording, peaks[start_idx:end_idx], process_id, output_file)
    
    print(f"Process {process_id}: Completed processing {end_idx - start_idx} peaks")


def process_peaks(recording, peaks, process_id, output_file, batch_size=128):
    """
    Process and save a chunk of data to its own HDF5 file.
    
    Args:
        recording: A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks: An array containing peaks information with unit ids.
        process_id: ID of the current process.
        output_file: Path to the output HDF5 file.
        batch_size: Number of samples to process in each batch.
    """
    # Create an HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create datasets for properties and traces
        f.create_dataset('properties', data=peaks)
        f.create_dataset('traces', shape=(len(peaks), 64, 192, 2), dtype='<f8')
        
        # Process the data in batches to save memory
        for start_idx in range(0, len(peaks), batch_size):
            end_idx = min(start_idx + batch_size, len(peaks))
            batch_times = peaks['sample_index'][start_idx:end_idx]
            
            # Process data for current batch
            batch_traces = np.array([get_trace_reshaped(recording, time) for time in batch_times])
            f['traces'][start_idx:end_idx] = batch_traces
            
            # Print progress
            print(f"Process {process_id}: [{end_idx}/{len(peaks)}]")


if __name__ == "__main__":
    args = parse_args()
    main(args)