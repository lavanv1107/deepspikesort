import datetime
import os

import numpy as np
import pandas as pd

from numba import jit
from tqdm import tqdm


def filter_peaks(recording, peaks):
    """
    Filters invalid peaks which are not centered between 31 and 33 frames.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks (obj): A peaks object returned by SpikeInterface's detect_peaks method.
 
    Returns:
        obj: A structured numpy array containing filtered peaks information.
    """    
    peaks = peaks[['sample_index', 'channel_index', 'amplitude']]
    
    # Create a boolean mask to identify rows within the specified range
    mask = (peaks['sample_index'] >= 31) & (peaks['sample_index'] <= recording.get_num_frames() - 33)

    peaks_filtered = peaks[mask]    
    
    return peaks_filtered
    

def match_peaks(peaks, spikes, channel_locations):
    """
    Match detected peaks with spikes from the NWB file.
    
    Here are the conditions for the matching process:
    - Spikes occurring within a range of 4 values from the sample_index of a peak are selected as possible matches.
    - Euclidean distance (channel location x, channel location y, sample_index) is calculated between a spike and the peak.
    - The closest match is found when the Euclidean distance is less than 100 μm, the least distance we set as a threshold.
    - Once we have iterated through all possible matches, the peak is assigned the unit ID of the matching spike.
    - If there is no spike match, the unit ID for that peak is set to -1.
    
    Args:
        peaks (array): A structured array containing peaks information with channel_index.
        spikes (array): A structured array containing spike information.
        channel_locations (array): A structured array containing channel_ind, channel_location_x, and channel_location_y.
 
    Returns:
        array: An array containing peaks information with unit ids.
    """
    # Create an array to store matches
    matches = -1 * np.ones(len(peaks), dtype='<i8')
    
    # Iterate through peaks_array using indices 
    for i, peak in tqdm(enumerate(peaks), total=len(peaks), desc="match peaks to spikes"): 
        # Find channel location for the current peak
        channel_idx = peak['channel_index']
        channel_mask = channel_locations['channel_index'] == channel_idx
        
        # Skip if channel not found
        if not np.any(channel_mask):
            continue
            
        # Get channel location for the peak
        peak_loc_x = channel_locations[channel_mask]['channel_location_x'][0]
        peak_loc_y = channel_locations[channel_mask]['channel_location_y'][0]
            
        # Find the starting index for possible matches using binary search
        start_idx = np.searchsorted(spikes['sample_index'], peak['sample_index'] - 4)
        
        # Set the least amount of distance between the current peak and spike channels
        least_distance = 100 # in um
        
        # Iterate through potential matching spikes
        for x in range(start_idx, len(spikes)):
            spike = spikes[x]
            
            # Exit the loop if spike sample_index exceeds the upper bound
            if spike['sample_index'] > peak['sample_index'] + 4:
                break
                
            # Get channel location for the spike
            spike_channel_idx = spike['channel_index']
            spike_channel_mask = channel_locations['channel_index'] == spike_channel_idx
            
            # Skip if channel not found
            if not np.any(spike_channel_mask):
                continue
                
            # Get channel location for the spike
            spike_loc_x = channel_locations[spike_channel_mask]['channel_location_x'][0]
            spike_loc_y = channel_locations[spike_channel_mask]['channel_location_y'][0]
            
            # Calculating the distance in space and time
            # This gives equal weight between 1 um and 1 time sample
            distance = np.sqrt(
                (spike_loc_x - peak_loc_x)**2 + 
                (spike_loc_y - peak_loc_y)**2 + 
                (spike['sample_index'] - peak['sample_index'])**2
            )
            
            # Set a unit_id for the peak if a spike is closest in distance to the current peak
            if distance < least_distance:
                matches[i] = spike['unit_index']
                least_distance = distance
                
    # Define a new dtype
    dt = [('sample_index', '<i8'), ('channel_index', '<i8'), ('amplitude', '<f8')] + [('unit_index', '<i8') ]
    
    # Create a new array with the new dtype
    peaks_matched = np.zeros(peaks.shape, dtype=np.dtype(dt))
    
    # Copy data from the old array to the new array
    for field_name in peaks.dtype.names:
        peaks_matched[field_name] = peaks[field_name]
        
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
