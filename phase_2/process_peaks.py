import os

import numpy as np
import pandas as pd

from numba import jit
from tqdm import tqdm


def filter_peaks(recording, peaks):
    """
    Filters peaks which do not fit in with the time frame format (centered between 31 and 33 frames) will be dropped.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks (obj): A peaks object returned by SpikeInterface's detect_peaks method.
 
    Returns:
        obj: A table containing peaks information.
    """
    # Columns to keep
    columns = ['sample_index', 'channel_index']

    # Define a new dtype with only the desired columns
    dtype = np.dtype([(name, peaks.dtype[name]) for name in columns])

    # Create a new array with the new dtype
    peaks_filtered = np.zeros(peaks.shape, dtype=dtype)

    # Copy data from the old array to the new array
    for name in dtype.names:
        peaks_filtered[name] = peaks[name]
    
    # Create a boolean mask to identify rows within the specified range
    mask = (peaks['sample_index'] >= 31) & (peaks['sample_index'] <= recording.get_num_frames() - 33)

    peaks_filtered = peaks_filtered[mask]
    
    return peaks_filtered
    
    
@jit
def match_peaks(peaks, spikes, channels):
    """
    There are peaks that were detected which are spikes. We can verify this using the spikes information within the NWB file.
    However, there are peaks which are not exact spike matches but they are very close and can be considered as such.
    
    Here are the conditions for the matching process:
    - Spikes occurring within a range of 4 values from the time of a peak are selected as possible matches.
    - Euclidean distance (channel location x, channel location y, time) is then calculated between a spike and the peak.
    - The closest match is found when the Euclidean distance is less than 100 μm, the least distance we set as a threshold.
    - Once we have iterated through all possible matches, the peak is assigned the unit ID of the matching spike.
    - If there is no spike match, the unit ID for that peak is set to -1.
    
    Once we have completed the spike matching process, a new peaks table will be created with a unit_id column.
 
    Args:
        peaks_table (obj): A table containing peaks information.
        spikes_table (obj): A table containing spike information.
        channels_table (obj): A table of channel locations.
 
    Returns:
        obj: An array containing peaks information with unit ids.
    """
    # Create an array to store matches
    matches = -1 * np.ones(peaks.shape[0], dtype=int)

    # Iterate through peaks_array using indices 
    for i, peak in tqdm(enumerate(peaks), total=peaks.shape[0], desc="matching peaks to spikes", dynamic_ncols=True):
        # Find the starting index for possible matches using binary search
        start_index = np.searchsorted(spikes['sample_index'], peak[0] - 4)

        # Get the location of the current peak's channel
        peak_loc = channels[peak[1]]

        # Set the least amount of distance between the current peak and spike channels
        least_distance = 100 # in um

        # Iterate through potential matching spikes
        for x in range(start_index, spikes.shape[0]):
            spike_time = spikes['sample_index'][x]

            # Exit the loop if spike_time exceeds the upper bound
            if spike_time > peak[0] + 4:
                break

            # Get the location of the current spike's channel
            spike_channel = spikes['channel_index'][x]
            spike_loc = channels[spike_channel]

            # Calculating the distance in space and time
            # This gives equal weight between 1 um and 1 time sample
            distance = np.sqrt((spike_loc[0] - peak_loc[0])**2 + 
               (spike_loc[1] - peak_loc[1])**2 + 
               (spike_time - peak[0])**2)

            # Set a unit_id for the peak if a spike is closest in distance to the current peak
            if distance < least_distance:
                matches[i]= spikes['unit_index'][x]
                least_distance = distance

    # Existing column names
    columns_peaks = list(peaks.dtype.names)

    # New order of columns including the new column
    order_matched = columns_peaks[:0] + ['unit_index'] + columns_peaks[0:]

    # Define a new dtype with columns in the new order
    dtype_matched = [(name, peaks.dtype[name]) if name in columns_peaks else (name, '<i8') for name in order_matched]

    # Create a new array with the new dtype
    peaks_matched = np.zeros(peaks.shape, dtype=dtype_matched)

    # Copy data from the old array to the new array
    for name in columns_peaks:
        peaks_matched[name] = peaks[name]
        
    # Fill the new column with matches
    peaks_matched['unit_index'] = matches

    return peaks_matched
    
    
def get_peaks_spikes(peaks_matched_table):
    """
    Creates a table of only spike matched peaks with unit ids.
 
    Args:
        peaks_matched_table (obj): A table containing peaks information with unit ids.
 
    Returns:
        obj: A table containing spike matched peaks.
    """
    peaks_spikes_table = peaks_matched_table[peaks_matched_table['unit_id'] > -1]
    peaks_spikes_table.reset_index(drop=True, inplace=True)
    
    return peaks_spikes_table


def get_peaks_noise(peaks_matched_table):
    """
    Creates a table of only non spike matched peaks with no unit ids.
 
    Args:
        peaks_matched_table (obj): A table containing peaks information with unit ids.
 
    Returns:
        obj: A table containing non spike macthed peaks.
    """
    peaks_noise_table = peaks_matched_table[peaks_matched_table['unit_id'] == -1]
    peaks_noise_table.drop(columns='unit_id', inplace=True)
    peaks_noise_table.reset_index(drop=True, inplace=True)

    return peaks_noise_table
