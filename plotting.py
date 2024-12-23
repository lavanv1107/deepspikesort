import itertools
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import HTML, display, clear_output

import preprocessing


def plot_trace_waveform(recording, sample_time, channels):
    """
    Plots waveforms at the specified time frame for multiple channels, each in its own subplot.
    
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        sample_time (int): A frame number when a sample occurred.
        channels (int or list): A channel number or a list of channel numbers.
    
    Returns:
        obj: A 2D plot of a waveform.
    """
    # Split channels into odd and even
    odd_channels = channels[channels % 2 == 1]
    even_channels = channels[channels % 2 == 0]
    
    # Set number of columns to 2 (one for odd channels, one for even channels)
    cols = 2
    num_plots_odd = len(odd_channels)
    num_plots_even = len(even_channels)
    
    # Calculate the number of rows needed for each side
    rows = max(num_plots_odd, num_plots_even)
    
    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(8, 3 * rows))  
    
    # Retrieve trace snippet for the given sample time
    trace_snippet = preprocessing.get_trace_snippet(recording, sample_time)
    
    # Plot waveforms for odd channels
    for i, channel in enumerate(odd_channels):
        axs[i, 0].plot(trace_snippet[:, channel])
        axs[i, 0].set_title(f'Channel {channel}')
    
    # Plot waveforms for even channels
    for i, channel in enumerate(even_channels):
        axs[i, 1].plot(trace_snippet[:, channel])
        axs[i, 1].set_title(f'Channel {channel}')
    
    # Disable unused subplots
    for i in range(num_plots_odd, rows):
        axs[i, 0].axis('off')
    
    for i in range(num_plots_even, rows):
        axs[i, 1].axis('off')

    # Set main title
    fig.suptitle(f'Sample Index: {sample_time}', fontsize=14)

    # Add one y-label and x-label for the whole plot
    fig.text(0.04, 0.5, 'action potential (μV)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 0.04, 'time (frames)', ha='center', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

    
def plot_trace_image(recording, sample_frame, columns):
    """
    Plots a 3D image of waveforms at the specified time frame and all channels.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        sample_frame (int): A frame number when a sample occurred.
        columns (str): 'single' for all channels in one column. 'double' for channels split into two columns.
 
    Yields:
        obj: A 3D image of waveforms.
    """
    if columns == 'single':
        trace_snippet = preprocessing.get_trace_snippet(recording, sample_frame)        
        trace_transposed = np.transpose(trace_snippet)
        
        vmin = trace_transposed.min()
        vmax = trace_transposed.max()

        plt.figure(figsize=(8, 6))
        plt.imshow(trace_transposed, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')

        # Set x and y labels for the plot
        plt.xlabel('time (frames)')
        plt.ylabel('channel index')

        # Add colorbar for the plot
        cb = plt.colorbar(orientation='vertical')

        plt.tight_layout()
        plt.show()
        
    if columns == 'double':
        trace_reshaped = preprocessing.get_trace_reshaped(recording, sample_frame)
        trace_transposed = np.transpose(trace_reshaped, (1, 0, 2))

        vmin = trace_transposed.min()
        vmax = trace_transposed.max()

        plt.figure(figsize=(8, 10))
        for i in range(trace_reshaped.shape[2]):
            plt.subplot(1, 2, i + 1)
            plt.imshow(trace_transposed[:, :, i], cmap='viridis', vmin=vmin, vmax=vmax)
        # Set x and y labels for the plot
        plt.text(0.5, 0.05, 'time (frames)', ha='center', va='center', transform=plt.gcf().transFigure)
        plt.text(0.01, 0.5, 'channel index', ha='center', va='center', rotation='vertical', transform=plt.gcf().transFigure)
        # Add colorbar for the plot
        cax = plt.axes([0.15, 0.95, 0.7, 0.03])  # [left, bottom, width, height]
        cb = plt.colorbar(cax=cax, orientation='horizontal')

        plt.show()
        
        
def plot_unit_image(recording, spikes, unit_id, columns, seed=0):
    unit = preprocessing.get_unit(spikes, unit_id)
    
    np.random.seed(seed)
    unit_spike = np.random.choice(unit)
    
    unit_spike_frame = unit_spike['time']
    
    plot_trace_image(recording, unit_spike_frame, columns)
    
    
def plot_unit_waveform(recording, spikes, unit_id, channel_id, all_waveforms=False, num_waveforms=10, seed=0):
    """
    Plots waveforms for a specific spike unit at its extremum channel.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        spikes (obj): An array containing spike information.
        unit_id (int): ID number of a unit.
        all_waveforms (bool): Condition to plot all spikes within the unit.
        num_waveforms (int): number of spikes to plot.
 
    Returns:
        obj: A 2D plot of waveforms.
    """
    sample_frames = preprocessing.get_unit_frames(spikes, unit_id)

    if all_waveforms:
        frames_to_plot = sample_frames

    else:
        if len(sample_frames) < num_waveforms:
            frames_to_plot = sample_frames
        else:
            np.random.seed(seed)  
            frames_to_plot = np.random.choice(sample_frames, num_waveforms)
            
    plt.figure()
    
    for frame in frames_to_plot:
        trace_snippet = preprocessing.get_trace_snippet(recording, frame)
        plt.plot(trace_snippet[:, channel_id])

    plt.xlabel('time (frames)')
    plt.ylabel('action potential (mV)')
    plt.title(f'Unit Index: {unit_id}\nChannel Index: {channel_id}')
    
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
        
