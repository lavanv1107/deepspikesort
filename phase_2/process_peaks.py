import datetime
import os

import numpy as np
import pandas as pd

from numba import jit
from tqdm import tqdm


def filter_peaks(recording, peaks, channels):
    """
    Filters peaks which do not fit in with the time frame format (centered between 31 and 33 frames) will be dropped.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks (obj): A peaks object returned by SpikeInterface's detect_peaks method.
 
    Returns:
        obj: A table containing peaks information.
    """
    # Define a new dtype 
    dt = peaks.dtype.descr[:-1] + [('peak_index', '<i8'), ('channel_location_x', '<i8'), ('channel_location_y', '<i8')]
    
    dt = [dt[idx] for idx in [3, 0, 1, 4, 5, 2]]

    # Create a new array with the new dtype
    peaks_filtered = np.empty(len(peaks), dtype=np.dtype(dt))
    
    peaks_filtered['peak_index'] = np.arange(0, len(peaks_filtered))

    # Copy data from the old array to the new array
    for descr in [dt[1], dt[2], dt[5]]:
        peaks_filtered[descr[0]] = peaks[descr[0]]   
        
    dt[1] = ('time', '<i8')
    peaks_filtered = peaks_filtered.astype(dt)
    
    # Create a boolean mask to identify rows within the specified range
    mask = (peaks_filtered['sample_index'] >= 31) & (peaks_filtered['sample_index'] <= recording.get_num_frames() - 33)

    peaks_filtered = peaks_filtered[mask]    
        
    channel_loc_dict = {channel['channel_index']: (channel['channel_location_x'], channel['channel_location_y']) for channel in channels}
    
    for i, peak in tqdm(enumerate(peaks_filtered), total=len(peaks_filtered), desc="add channel locations"):
        channel_loc_x, channel_loc_y = channel_loc_dict.get(peak['channel_index'], ('Unknown', 'Unknown'))  # Handles missing IDs
        peaks_filtered[i]['channel_location_x'] = channel_loc_x
        peaks_filtered[i]['channel_location_y'] = channel_loc_y
    
    return peaks_filtered
    
    

def match_peaks(peaks, spikes):
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
    matches = -1 * np.ones(len(peaks), dtype='<i8')

    # Iterate through peaks_array using indices 
    for i, peak in tqdm(enumerate(peaks), total=len(peaks), desc="match peaks to spikes"): 
            
        # Find the starting index for possible matches using binary search
        start_idx = np.searchsorted(spikes['time'], peak['time'] - 4)

        # Set the least amount of distance between the current peak and spike channels
        least_distance = 100 # in um

        # Iterate through potential matching spikes
        for x in range(start_idx, len(spikes)):
            spike = spikes[x]

            # Exit the loop if spike_time exceeds the upper bound
            if spike['time'] > peak['time'] + 4:
                break

            # Calculating the distance in space and time
            # This gives equal weight between 1 um and 1 time sample
            distance = np.sqrt(
                (spike['channel_location_x'] - peak['channel_location_x'])**2 + 
                (spike['channel_location_y'] - peak['channel_location_y'])**2 + 
                (spike['time'] - peak['time'])**2
            )

            # Set a unit_id for the peak if a spike is closest in distance to the current peak
            if distance < least_distance:
                matches[i]= spike['unit_index']
                least_distance = distance

    # Define a new dtype
    dt = peaks.dtype.descr + [('unit_index', '<i8')]

    # Create a new array with the new dtype
    peaks_matched = np.zeros(peaks.shape, dtype=np.dtype(dt))

    # Copy data from the old array to the new array
    for column in dt[:-1]:
        peaks_matched[column[0]] = peaks[column[0]]
        
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
