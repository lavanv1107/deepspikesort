import argparse
import os

import numpy as np
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Combine individual HDF5 files into a single file.")
    parser.add_argument('recording_id', type=str, help="ID of the recording.")
    parser.add_argument('num_tasks', type=int, default=1, help="Number of tasks used in processing.")
    return parser.parse_args()


def main(args):
    data_folder = f'data/{args.recording_id}'
    
    # Load peak data and channel locations
    peaks = np.load(os.path.join(data_folder, "peaks/peaks_test.npy"))
    channel_locations = np.load(os.path.join(data_folder, "channel_locations.npy"))
    
    # Combine all the peak files
    combine_peaks(peaks, channel_locations, args.num_tasks, data_folder)


def combine_peaks(peaks, channel_locations, num_tasks, data_folder):
    """
    Combine all individual HDF5 files into a single file.
    
    Args:
        peaks: The original peaks array.
        channel_locations: Array containing channel location information.
        num_tasks: Total number of processes that were run.
        data_folder: Path to the data folder.
    """
    # Create the final combined file
    combined_file = os.path.join(data_folder, "peaks/peaks.h5")
    with h5py.File(combined_file, 'w') as combined:
        combined.create_dataset('channel_locations', data=channel_locations)
        
        # Create datasets for properties and traces
        combined.create_dataset('properties', shape=(len(peaks),), dtype=peaks.dtype)
        combined.create_dataset('traces', shape=(len(peaks), 64, 192, 2), dtype='<f8')
        
        # Copy data from each file into the combined file
        current_row = 0
        for i in range(num_tasks):
            output_file = os.path.join(data_folder, f"peaks/peaks_{i}.h5")
            
            with h5py.File(output_file, 'r') as f:
                num_rows = len(f['properties'])
                combined['properties'][current_row:current_row + num_rows] = f['properties'][:]
                combined['traces'][current_row:current_row + num_rows] = f['traces'][:]
                current_row += num_rows
                print(f"Added {num_rows} rows from process {i}")
        
        print(f"Successfully combined all data, total {current_row} rows written to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)