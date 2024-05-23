import os

import numpy as np
import pandas as pd

import spikeinterface.full as si


def extract_channels(recording):
    """
    Creates an array of channel locations on the probe. 
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
 
    Returns:
        obj: An array of channel locations.
    """
    channel_locations = recording.get_channel_locations()

    # Define the structured data type
    dtype = [('channel_loc_x', '<f8'), ('channel_loc_y', '<f8')]

    # Create an empty array with the structured dtype
    channels = np.empty(len(channel_locations), dtype=dtype)

    # Assign values from the original array to the structured array
    channels['channel_loc_x'] = channel_locations[:, 0]
    channels['channel_loc_y'] = channel_locations[:, 1]

    return channels


def extract_spikes(sorting, waveform):
    """
    Creates an array of each spike event including the frame and channel at which its peak occurred. 
 
    Args:
        sorting (obj): A SortingExtractor object created from an NWB file using SpikeInterface.
        waveform (obj): A WaveformExtractor object created from a RecordingExtractor and SortingExtractor object.
 
    Returns:
        obj: A numpy array of spike events.
    """
    extremum_channels = si.get_template_extremum_channel(waveform, outputs="index")
    
    spikes = sorting.to_spike_vector(extremum_channel_inds = extremum_channels)
    
    # Columns to keep
    columns_filtered = ['unit_index', 'sample_index', 'channel_index']

    # Define a new dtype with only the desired columns
    dtype_filtered = np.dtype([(name, spikes.dtype[name]) for name in columns_filtered])

    # Create a new array with the new dtype
    spikes_filtered = np.zeros(spikes.shape, dtype=dtype_filtered)

    # Copy data from the old array to the new array
    for name in dtype_filtered.names:
        spikes_filtered[name] = spikes[name]

    return spikes_filtered


def create_noise(recording, spikes, num_samples=100000):
    """
    Creates a noise unit by selecting random samples from the recording that do not overlap with spike events.

    Args:
        recording: A RecordingExtractor object that provides access to the recording data.
        spikes (np.ndarray): An array containing spike information, specifically the sample indices of spikes.

    Returns:
        np.ndarray: A structured array containing noise samples with unit index set to -1 and sample indices of the noise.
    """
    # Generate a range of sample indices, excluding the edges 
    samples_range = np.arange(31, recording.get_num_frames() - 33)
    
    # Create a mask to exclude samples that are present in the spikes array
    mask = ~np.isin(samples_range, spikes['sample_index'])
    
    # Select the samples that are not in the spikes array
    noise_samples = samples_range[mask]
    
    # Randomly select samples from the noise samples 
    selected_samples = np.random.choice(noise_samples, size=num_samples, replace=False)
    
    # Define the structured data type for the noise array
    dtype = [('unit_index', '<i8'), ('sample_index', '<i8')]

    # Create an empty array with the defined structured dtype
    noise = np.empty(len(selected_samples), dtype=dtype)
    
    # Populate the unit_index with -1 to indicate noise
    noise['unit_index'] = -1 
    
    # Assign the selected sample indices to sample_index
    noise['sample_index'] = selected_samples
    
    return noise


def get_unit(spikes, unit_ids):
    """
    Creates an array of spike events for one or more units.

    Args:
        spikes (obj): An array of spike events.
        unit_ids (int or list of int): A single unit ID or a list of unit IDs.

    Returns:
        obj: An array of spike events for the specified unit(s).
    """
    if isinstance(unit_ids, int):
        # If a single unit ID is provided, convert it to a list
        unit_ids = [unit_ids]
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise TypeError("unit_ids should be an int or a list/array of ints.")
    
    unit = spikes[np.isin(spikes['unit_index'], unit_ids)]
    return unit
    
    
def get_unit_frames(spikes, unit_id):
    """
    Returns all sample frames of a single unit.
 
    Args:
        spikes (obj): An array of spike events.
        unit_id (int): ID number of a unit.
 
    Returns:
        sample_frames (obj): A list of sample frames for a single unit.
    """
    unit = get_unit(spikes, unit_id)
    sample_frames = unit['sample_index']
    return sample_frames


def get_trace_snippet(recording, sample_frame):
    """
    Spikes generally occur for 2ms (or 64 frames).
    SpikeInterface's get_traces function retrieves a trace of action potentials in all channels within a specficied time frame.
    This returns a 64 frame trace centered in time on a specified sample frame in order to capture a spike.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        sample_frame (int): A frame number when a sample occurred.
 
    Returns:
        obj: A 2D numpy array (64, 384) of action potentials.
    """
    trace_snippet = recording.get_traces(start_frame=sample_frame - 31, end_frame=sample_frame + 33)
    return trace_snippet


def get_trace_reshaped(recording, sample_frame):
    """
    Channels on a Neuropixels probe are arranged in two columns in a checkerboard pattern. 
    There are 4 vertical channel locations - (16,0) in the first column and (48, 32) in the second column. 
    From the first to last channel, locations follow in this order: 16, 48, 0, 32.
    Knowing this, the trace can be reshaped in order to best represent this spatial property.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        sample_frame (int): A frame number when a sample occurred.
 
    Returns:
        obj: A 3D numpy array (64, 192, 2) of action potentials.
    """
    trace_snippet = get_trace_snippet(recording, sample_frame)
    trace_reshaped = np.dstack((
            trace_snippet[:, ::2],
            trace_snippet[:, 1::2]
        ))
    return trace_reshaped
