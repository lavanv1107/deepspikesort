import numpy as np
import pandas as pd

import os

import spikeinterface.full as si


def channel_slice_electricalseriesap(recording):
    """
    Extracellular recordings can contain both action potential (AP) and low frequency (LF) electrical series.
    SpikeInterface creates a RecordingExtractor object with 768 channels 
    This extracts a slice of a recording with its AP electrical series.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
 
    Returns:
        obj: A sliced RecordingExtractor object with AP channels.
    """
    channel_ids = recording.get_channel_ids()
    channel_ids_slice = channel_ids[0:384]
    recording_slice = recording.channel_slice(channel_ids=channel_ids_slice)
    return recording_slice


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


def get_unit(spikes, unit_id):
    """
    Creates an array of spike events for a single unit. 
 
    Args:
        spikes (obj): An array of spike events.
        unit_id (int): ID number of a unit.
 
    Returns:
        obj: An array of spike events for a single unit.
    """
    unit = spikes[spikes['unit_index'] == unit_id]
    return unit
    
    
def get_unit_frames_and_channel(spikes, unit_id):
    """
    Returns all sample frames and extremum channel of a single unit.
 
    Args:
        spikes (obj): An array of spike events.
        unit_id (int): ID number of a unit.
 
    Returns:
        sample_frames (obj): A list of sample frames for a single unit.
        extremum_channel (int): ID number of extremum channel for a single unit.
    """
    unit = get_unit(spikes, unit_id)
    sample_frames = unit['sample_index'].to_list()
    extremum_channel = unit['channel_index'].unique()[0]
    return sample_frames, extremum_channel


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
