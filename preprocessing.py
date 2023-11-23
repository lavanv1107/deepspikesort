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
    Creates a table of channel locations on the probe. 
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
 
    Returns:
        obj: A table of channel locations.
    """
    channel_ids = recording.get_channel_ids()
    channel_locations = recording.get_channel_locations()

    channels_array = np.hstack((channel_ids.reshape(-1, 1), channel_locations))

    channel_columns = ['channel_id', 'channel_loc_x', 'channel_loc_y']

    channels_table = pd.DataFrame(channels_array, columns=channel_columns)
    channels_table['channel_id'] = channels_table['channel_id'].astype(str)
    channels_table['channel_loc_x'] = channels_table['channel_loc_x'].astype(float)
    channels_table['channel_loc_y'] = channels_table['channel_loc_y'].astype(float)

    return channels_table


def extract_spikes(sorting, waveform):
    """
    Creates a table of each spike event including the frame and channel at which its peak occurred. 
 
    Args:
        sorting (obj): A SortingExtractor object created from an NWB file using SpikeInterface.
        waveform (obj): A WaveformExtractor object created from a RecordingExtractor and SortingExtractor object.
 
    Returns:
        obj: A table of spike events.
    """
    spikes_table = pd.DataFrame({'unit_id': sorting.to_spike_vector()['unit_index'],
                                     'peak_frame': sorting.to_spike_vector()['sample_index']})

    spikes_table['unit_id'] = spikes_table['unit_id'].astype(int)

    # Create a new column and map values from the dictionary based on matching keys
    spikes_table['peak_channel'] = spikes_table['unit_id'].map(si.get_template_extremum_channel(waveform, outputs="index"))

    return spikes_table


def get_unit(spikes_table, unit_id):
    """
    Creates a table of spike events for a single unit. 
 
    Args:
        spikes_table (obj): A table of spike events.
        unit_id (int): ID number of a unit.
 
    Returns:
        obj: A table of spike events for a single unit.
    """
    unit_table = spikes_table[spikes_table['unit_id'] == unit_id]
    unit_table.reset_index(drop=True, inplace=True)
    return unit_table
    
    
def get_unit_frames_and_channel(spikes_table, unit_id):
    """
    Returns all peak frames and peak channel of a single unit.
 
    Args:
        spikes_table (obj): A table of spike events.
        unit_id (int): ID number of a unit.
 
    Returns:
        peak_frames (obj): A list of peak frames for a single unit.
        peak_channel (int): ID number of peak channel for a single unit.
    """
    unit_table = get_unit(spikes_table, unit_id)
    peak_frames = unit_table['peak_frame'].to_list()
    peak_channel = unit_table['peak_channel'].unique()[0]
    return peak_frames, peak_channel


def get_trace_snippet(recording, peak_frame):
    """
    Spikes generally occur for 2ms (or 64 frames).
    SpikeInterface's get_traces function retrieves a trace of action potentials in all channels within a specficied time frame.
    This returns a 64 frame trace centered in time on a specified peak frame in order to capture a spike.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peak_frame (int): A frame number when a peak occurred.
 
    Returns:
        obj: A 2D numpy array (64, 384) of action potentials.
    """
    trace_snippet = recording.get_traces(start_frame=peak_frame - 31, end_frame=peak_frame + 33)
    return trace_snippet


def get_trace_reshaped(recording, peak_frame):
    """
    Channels on a Neuropixels probe are arranged in two columns in a checkerboard pattern. 
    There are 4 vertical channel locations - (16,0) in the first column and (48, 32) in the second column. 
    From the first to last channel, locations follow in this order: 16, 48, 0, 32.
    Knowing this, the trace can be reshaped in order to best represent this spatial property.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peak_frame (int): A frame number when a peak occurred.
 
    Returns:
        obj: A 3D numpy array (64, 192, 2) of action potentials.
    """
    trace_snippet = get_trace_snippet(recording, peak_frame)
    trace_reshaped = np.dstack((
            trace_snippet[:, ::2],
            trace_snippet[:, 1::2]
        ))
    return trace_reshaped
