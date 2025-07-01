"""
For our experiments, we parallelize the creation of our peaks dataset (traces from all channels around
detected peaks) into N tasks. Each task generates an HDF5 file with two datasets: "properties" and
"traces". This script combines the datasets from those N HDF5 files into one HDF5 file named 
"peaks.h5" and adds another dataset, "channel_locations". 
"""

import argparse
import glob
import os
import re

import numpy as np
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Combine individual HDF5 files into a single file.")
    parser.add_argument('recording_id', type=str, help="ID of the recording.")
    return parser.parse_args()


def main(args):
    data_folder = f'data/{args.recording_id}'
    peaks_folder = os.path.join(data_folder, "peaks")
    
    # Load the original peak data and channel locations
    peaks = np.load(os.path.join(peaks_folder, "peaks.npy"))
    channel_locations = np.load(os.path.join(data_folder, "channel_locations.npy"))
    
    # Find all peaks files automatically
    peaks_files = find_peaks_files(peaks_folder)
    
    # Combine all the peak files
    combine_peaks_files(peaks, channel_locations, peaks_files, data_folder)


def find_peaks_files(folder):
    """
    Automatically find all peaks_X.h5 files in the given folder.
    
    Args:
        folder: Path to the folder containing peak files.
        
    Returns:
        List of paths to peak files.
    """
    # Find all peaks_*.h5 files
    peaks_files = glob.glob(os.path.join(folder, "peaks_*.h5"))
    
    # Sort files numerically by their indices
    peaks_files.sort(key=lambda f: int(re.search(r'peaks_(\d+)\.h5$', f).group(1)))
    
    return peaks_files


def combine_peaks_files(peaks, channel_locations, peaks_files, data_folder):
    """
    Combine all individual HDF5 files into a single file.
    
    Args:
        peaks: The original peaks array.
        channel_locations: Array containing channel location information.
        peaks_files: List of paths to peak files.
        data_folder: Path to the data folder.
    """
    print(f"Combining results from {len(peaks_files)} files...")
    
    # Create the final combined file
    output_file = os.path.join(data_folder, "peaks/peaks.h5")
    with h5py.File(output_file, 'w') as combined:
        combined.create_dataset('channel_locations', data=channel_locations)
        
        # Create datasets for properties and traces
        combined.create_dataset('properties', shape=(len(peaks),), dtype=peaks.dtype)
        combined.create_dataset('traces', shape=(len(peaks), 64, 192, 2), dtype='<f8')
        
        # Copy data from each file into the combined file
        current_row = 0
        for file_path in peaks_files:
            print(f"Loading data from {file_path}")
            
            with h5py.File(file_path, 'r') as f:
                num_rows = len(f['properties'])
                combined['properties'][current_row:current_row + num_rows] = f['properties'][:]
                combined['traces'][current_row:current_row + num_rows] = f['traces'][:]
                current_row += num_rows
                print(f"Added {num_rows} rows from {os.path.basename(file_path)}")
        
        print(f"Successfully combined all data, total {current_row} rows written to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
