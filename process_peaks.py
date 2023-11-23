import os

import numpy as np
import pandas as pd

from numba import jit
from tqdm import tqdm


def save_peaks(peaks, folder):
    """
    Saves peaks object to specified folder path in disk.
 
    Args:
        peaks (obj): A peaks object returned by SpikeInterface's detect_peaks method.
        folder (str): A folder path name.
    """
    os.mkdir(folder)
    np.save(os.path.join(folder, 'peaks.npy'), peaks)

    
def load_peaks(folder):
    """
    Loads peaks object from specified folder path in disk.
 
    Args:
        folder (str): A folder path name.
 
    Returns:
        obj: A peaks object loaded from disk. 
    """
    return np.load(os.path.join(folder, 'peaks.npy'))


def extract_peaks(recording, peaks):
    """
    Creates a table using information from a peaks object.
    Peaks which do not fit in with the time frame format (centered between 31 and 33 frames) will be dropped.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks (obj): A peaks object returned by SpikeInterface's detect_peaks method.
 
    Returns:
        obj: A table containing peaks information.
    """
    peaks_table = pd.DataFrame(peaks)
    
    peaks_table.rename(columns={'sample_index': 'peak_frame', 'channel_index':'peak_channel'}, inplace=True)
    peaks_table.drop(columns=['amplitude','segment_index'], inplace=True)

    # Create a boolean mask to identify rows within the specified range
    mask = (peaks_table['peak_frame'] >= 31) & (peaks_table['peak_frame'] <= recording.get_num_frames() - 33)

    peaks_table = peaks_table[mask]
    peaks_table = peaks_table.reset_index(drop=True)
    
    return peaks_table
    
    
@jit
def match_peaks(peaks_table, spikes_table, channels_table):
    """
    There are peaks that were detected which are spikes. We can verify this using the spikes information within the NWB file.
    However, there are peaks which are not exact spike matches but they are very close and can be considered as such.
    
    Here are the conditions for the spike selection process:
    - We first search for possible spike matches in the spikes table for a peak within a range of 4 values.
    - After we find possible matches, we calculate the Euclidean distance (time, channel location x, channel location y) to find the closest spike match for the peak.
    - Once we have iterate through all possible spike matches, the one with the least distance will be selected and the peak will be given the spike's unit id.
    - If there is no spike match, the unit id for that peak will be set to -1
    
    Once we have completed the spike matching process, a new peaks table will be created with a unit_id column.
 
    Args:
        peaks_table (obj): A table containing peaks information.
        spikes_table (obj): A table containing spike information.
        channels_table (obj): A table of channel locations.
 
    Returns:
        obj: A table containing peaks information with unit ids.
    """
    # Convert peaks and spikes data to numpy arrays
    peaks_array = peaks_table[['peak_frame', 'peak_channel']].values
    spikes_array = spikes_table[['unit_id', 'peak_frame', 'peak_channel']].values
    channels_array = channels_table[['channel_loc_x', 'channel_loc_y']].values

    # Create an array to store matches
    matches = -1 * np.ones(peaks_array.shape[0], dtype=int)

    # Iterate through peaks_array using indices 
    for i, peak in tqdm(enumerate(peaks_array), total=peaks_array.shape[0], desc="matching peaks", dynamic_ncols=True):
        # Find the starting index for possible matches using binary search
        start_index = np.searchsorted(spikes_array[:, 1], peak[0] - 4)

        # Get the location of the current peak's channel
        peak_loc = channels_array[peak[1]]

        # Set the least amount of distance between the current peak and spike channels
        least_distance = 100 # in um

        # Iterate through potential matching spikes
        for x in range(start_index, spikes_array.shape[0]):
            spike_time = spikes_array[x, 1]

            # Exit the loop if spike_time exceeds the upper bound
            if spike_time > peak[0] + 4:
                break

            # Get the location of the current spike's channel
            spike_loc = channels_array[spikes_array[x, 2]]

            # Calculating the distance in space and time
            # This gives equal weight between 1 um and 1 time sample
            distance = np.sqrt((spike_loc[0] - peak_loc[0])**2 + 
               (spike_loc[1] - peak_loc[1])**2 + 
               (spike_time - peak[0])**2)

            # Set a unit_id for the peak if a spike is closest in distance to the current peak
            if distance < least_distance:
                matches[i]= spikes_array[x, 0]
                least_distance = distance

    # Create the matched table
    peaks_matched_table = pd.DataFrame(peaks_array, columns=['peak_frame', 'peak_channel'])
    peaks_matched_table['unit_id'] = matches  # Add 'unit_id' column
    peaks_matched_table.insert(0, 'unit_id', peaks_matched_table.pop('unit_id'))

    return peaks_matched_table
    
    
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
