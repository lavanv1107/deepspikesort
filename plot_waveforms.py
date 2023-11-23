import matplotlib.pyplot as plt
import numpy as np
import random
import preprocessing


def plot_trace_waveform(recording, frame, channel):
    """
    Plots a waveform at the specified time frame and channel.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        frame (int): A frame number.
        channel (int): A channel number.
 
    Returns:
        obj: A 2D plot of a waveform.
    """
    trace_snippet = preprocessing.get_trace_snippet(recording, frame)
    
    plt.figure()

    plt.plot(trace_snippet[:, channel])
    
    plt.show()

    
def plot_trace_image(recording, frame):
    """
    Plots a 3D image of waveforms at the specified time frame and all channels.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        frame (int): A frame number.
 
    Yields:
        obj: A 3D image of waveforms.
    """
    trace_reshaped = preprocessing.get_trace_reshaped(recording, frame)
    trace_transposed = np.transpose(trace_reshaped, (1, 0, 2))

    vmin = trace_transposed.min()
    vmax = trace_transposed.max()

    plt.figure(figsize=(8, 9))
    for i in range(trace_reshaped.shape[2]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(trace_transposed[:, :, i], cmap='viridis', vmin=vmin, vmax=vmax)
    # Set x and y labels for the plot
    plt.text(0.5, 0.01, 'time (frames)', ha='center', va='center', transform=plt.gcf().transFigure)
    plt.text(0.01, 0.5, 'channel', ha='center', va='center', rotation='vertical', transform=plt.gcf().transFigure)
    # Add colorbar for the plot
    cax = plt.axes([0.15, 0.95, 0.7, 0.03])  # [left, bottom, width, height]
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    
    plt.show()
    
    
def plot_unit_waveform(recording, spikes_table, unit_id, all_waveforms=False, num_waveforms=10, seed=None):
    """
    Plots waveforms for a specific spike unit at its extremum channel.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        spikes_table (obj): A table containing spike information.
        unit_id (int): ID number of a unit.
        all_waveforms (bool): Condition to plot all spikes within the unit.
        num_waveforms (int): number of spikes to plot.
 
    Returns:
        obj: A 2D plot of waveforms.
    """
    peak_frames, peak_channel = preprocessing.get_unit_frames_and_channel(spikes_table, unit_id)

    if all_waveforms:
        peak_frames_to_plot = peak_frames

    else:
        if len(peak_frames) < num_waveforms:
            peak_frames_to_plot = peak_frames
        else:
            if seed is not None:
                random.seed(seed)
            peak_frames_to_plot = random.sample(peak_frames, num_waveforms)
            
    plt.figure()
    
    for peak_frame in peak_frames_to_plot:
        trace_snippet = preprocessing.get_trace_snippet(recording, peak_frame)
        plt.plot(trace_snippet[:, peak_channel])

    plt.xlabel('time (frames)')
    plt.title(f'Unit ID: {unit_id}\nPeak Channel: {peak_channel}')
    
    plt.show()
    
    
def plot_peak_waveform(recording, peaks_noise_table, start_idx, end_idx):
    """
    Plots waveforms for peaks within a specified range.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        peaks_noise_table (obj): A table containing peaks which are noise.
        start_idx (int): A start index number.
        end_idx (int): An end index number.
 
    Returns:
        obj: 2D plots of different waveforms.
    """
    for idx in range(start_idx, end_idx+1):
        peak_frame = peaks_noise_table.loc[idx, 'peak_frame']
        peak_channel = peaks_noise_table.loc[idx, 'peak_channel']
        plot_trace_waveform(recording, peak_frame, peak_channel)