"""

Import required libraries

"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random

import os
import pickle

from numba import jit
import multiprocessing as mp
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset

import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

import spikeinterface as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

"""

Preprocessing functions

"""

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
    spikes_table = pd.DataFrame({'unit_id': sorting.get_all_spike_trains()[0][1],
                                     'peak_frame': sorting.get_all_spike_trains()[0][0]})

    spikes_table['unit_id'] = spikes_table['unit_id'].astype(int)

    # Create a new column and map values from the dictionary based on matching keys
    spikes_table['peak_channel'] = spikes_table['unit_id'].map(si.get_template_extremum_channel(waveform, outputs="index"))

    return spikes_table

"""

Trace functions

"""

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

"""

Plotting functions

"""

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
    trace_snippet = get_trace_snippet(recording, frame)
    
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
    trace_reshaped = get_trace_reshaped(recording, frame)
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
    
def plot_unit_waveform(recording, spikes_table, unit_id, all_waveforms=False, num_waveforms=10):
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
    peak_frames, peak_channel = get_unit_frames_and_channel(spikes_table, unit_id)

    if all_waveforms:
        peak_frames_to_plot = peak_frames

    else:
        if len(peak_frames) < num_waveforms:
            peak_frames_to_plot = peak_frames
        else:
            peak_frames_to_plot = random.sample(peak_frames, num_waveforms)
            
    plt.figure()
    
    for peak_frame in peak_frames_to_plot:
        trace_snippet = get_trace_snippet(recording, peak_frame)
        plt.plot(trace_snippet[:, peak_channel])

    plt.xlabel('time (frames)')
    plt.title(f'Unit ID: {unit_id}\nPeak Channel: {peak_channel}')
    
    plt.show()
    
def plot_peak_waveform(recording, peaks_noise_table, start_idx, end_idx):
    """
    Plots waveforms for peaks which are noise within a specified range.
 
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

"""

Peaks

"""

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
    
    peaks_table.rename(columns={'sample_ind': 'peak_frame', 'channel_ind':'peak_channel'}, inplace=True)
    peaks_table.drop(columns=['amplitude','segment_ind'], inplace=True)

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

"""

Dataset functions

"""

def process_trace(recording, folder, frame):
    """
    Creates a trace of a specified frame and saves it to disk.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        folder (str): A folder path name.
        frame (int): A frame number.
    """
    trace_reshaped = get_trace_reshaped(recording, frame)
    trace_file = f"frame_{frame}.npy"
    np.save(os.path.join(folder, trace_file), trace_reshaped)

def process_unit(table, folder, unit_id, num_processes=128, batch_size=1000, all_frames=False, num_frames=0):
    """
    This creates traces for a specified unit's frames and saves them to disk.
    It creates a folder with a unit's id number. Traces will be created for all frames belonging to that unit and saved to the folder path.
    
    In order to speed up this I/O process, we can utilize multiprocessing as well as batches. 
    The number of processes can be set according to the number of cores available. 
    The batch size can be set according to how much memory is available.
    
    You can either use all frames belonging to a unit or you can set the number of frames to be used.
 
    Args:
        table (obj): A table containing entries with a unit id and frame.
        folder (str): A folder path name.
        unit_id (int): A spike unit's ID number.
        num_processes (int): number of processes for multiprocessing.
        batch_size (int): number of traces to process per batch.
        all_frames (bool): condition to use all frames belonging to a unit.
        num_frames (int): number of frames to use for a unit.
    """
    with mp.Pool(processes=num_processes) as pool:
        unit_folder = os.path.join(folder, f'unit_{unit_id}')
        if not os.path.exists(unit_folder):
            os.mkdir(unit_folder)

        unit_table = get_spike_unit(table, unit_id)
        unit_frames = unit_table.iloc[:, 1].to_list()

        if all_frames:
            num_frames = len(unit_table)

        for i in range(0, num_frames, batch_size):
            frames_batch = unit_frames[i:i+batch_size]
            folder_frames_batch = [(unit_folder, frame) for frame in frames_batch]
            pool.starmap(process_trace, tqdm(folder_frames_batch,
                                             total=len(folder_frames_batch),
                                             desc='processing batch',
                                             dynamic_ncols=True))

class TensorDataset(Dataset):
    """
    A custom PyTorch Dataset class which converts trace images in the image dataset into tensors and attaches labels to them.
 
    Attributes:
        Dataset (class): PyTorch's Dataset class.
    """
    def __init__(self, folder_path, folder_labels, folder_labels_map, num_spike_images=1000, num_noise_images=100000):
        """
        Args:
            folder_path (str): The folder path name of the image dataset.
            folder_labels (list): A list containing folder names of image dataset.
            folder_labels_map (dict): A dictionary mapping each folder name to a value.
            num_spike_images (int): Number of images used for spikes per spike unit.
            num_noise_images (int): Number of images used for noise.
        """
        self.folder_path = folder_path
        self.labels = folder_labels
        self.labels_map = folder_labels_map
        self.image_paths = self.get_image_paths(num_spike_images, num_noise_images)

    def get_image_paths(self, num_spike_images, num_noise_images):
        """
        Creates a list of path names of all images in the image dataset.

        Args:
            num_spike_images (int): Number of images used for spikes per spike unit.
            num_noise_images (int): Number of images used for noise.

        Returns:
            obj: A list of image path names.
        """
        image_paths = []
        # Iterate over the subfolders
        for folder in self.labels:
            folder_path = os.path.join(self.folder_path, folder)
            # Get all the numpy files in the subfolder
            folder_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
            # Add the full path of each file to the list
            if folder == 'noise':
                # Select a number of files from the noise folder
                image_paths.extend([os.path.join(folder_path, file) for file in folder_files])
            else:
                # Select a number of files from each spike unit folder
                image_paths.extend([os.path.join(folder_path, file) for file in folder_files])
        return image_paths

    def __len__(self):
        """
        Checks the number of images in the image dataset.

        Returns:
            int: The size of the image dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset, converts it into a tensor and attaches a label to it from labels_map which it belongs to.

        Args:
            int: Index number of image in the image dataset.

        Returns:
            image (obj): The size of the image dataset.
            label (int): The label of the image.
        """
        # Load the numpy file as a grayscale image tensor
        image = torch.from_numpy(np.load(self.image_paths[idx])).unsqueeze(0).float()
        # Extract the folder name from the file path
        folder_name = os.path.dirname(self.image_paths[idx])
        # Extract the label from the folder name
        label = folder_name.split(os.sep)[-1]  # Extract the last folder name
        # Assign the numerical label based on the label map
        label = self.labels_map[label]
        return image, label

"""

CNN functions

"""

class TrainModel():
    """
    A PyTorch-based class for training and validating a model.
    """
    def __init__(self, train_dataloader, test_dataloader, device, loss_fn, optimizer):
        """
        Args:
            train_dataloader (obj): The dataloader object for the training set.
            test_dataloader (obj): The dataloader object for the testing set.
            device (str): The name of the device to run training/testing.
            loss_fn (class): A loss function class provided by PyTorch.
            optimizer (obj): A PyTorch optimizer object created based on the model's parameters.
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    # Create training function
    def train(self, model):
        """
        This trains the model and prints the current loss after every batch.

        Args:
            model (obj): A PyTorch model.
        """
        # Start training the model
        model.train()

        # Get the total number of samples in the training dataset
        size = len(self.train_dataloader.dataset)

        # Iterate through each batch in the training dataloader
        for batch, (X, Y) in enumerate(self.train_dataloader):

            # Move input data and labels to the specified device (e.g., GPU)
            X = X.to(self.device)
            Y = Y.to(self.device)

            # Zero out the gradients in the model's parameters
            self.optimizer.zero_grad()

            # Forward pass: Compute predictions using the model
            pred = model(X)

            # Compute the loss between the predictions and the ground truth labels
            loss = self.loss_fn(pred, Y)

            # Backpropagation: Compute gradients of the loss with respect to model parameters
            loss.backward()

            # Update the model's parameters using the optimizer
            self.optimizer.step()

            # Print the loss and progress every 100 batches
            if batch % 100 == 0:
                loss_value = loss.item()
                current = batch * len(X)
                print(f'loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]')
                
    # Create testing function
    def test(self, model):
        """
        This tests/validates the model and prints the model's accuracy and average loss for an epoch.
        It also returns the true labels and predicted labels for an epoch as well.

        Args:
            model (obj): A PyTorch model.
            
        Returns:
            test_loss (int): The average test loss for an epoch.
            true_labels (obj): A list of the actual true labels being used for training.
            predicted_labels (obj): A list of the model's predicted labels when validating.
        """
        # Get the total number of samples in the test dataset
        size = len(self.test_dataloader.dataset)

        # Set the model to evaluation mode (no gradient updates during evaluation)
        model.eval()

        # Initialize variables to keep track of test loss and correct predictions
        test_loss, correct = 0, 0

        # Lists to store true labels and predicted labels for later analysis
        true_labels = []
        predicted_labels = []

        # Disable gradient computation for efficiency during evaluation
        with torch.no_grad():
            # Iterate through each batch in the test dataloader
            for batch, (X, Y) in enumerate(self.test_dataloader):

                # Move input data and labels to the specified device (e.g., GPU)
                X, Y = X.to(self.device), Y.to(self.device)

                # Forward pass: Compute predictions using the model
                pred = model(X)

                # Compute the test loss and add it to the running total
                test_loss += self.loss_fn(pred, Y).item()

                # Count correct predictions by comparing predicted labels to true labels
                correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

                # Store the true labels and predicted labels for further analysis
                true_labels.extend(Y.tolist())
                predicted_labels.extend(pred.argmax(1).tolist())

        # Calculate the average test loss and accuracy
        test_loss /= size
        correct /= size

        # Print the test results (accuracy and average loss)
        print(f'\nTest Error:\n- Accuracy: {(100 * correct):>0.1f}%\n- Average Loss: {test_loss:>8f}\n')

        # Return the test loss, true labels, and predicted labels for external analysis
        return test_loss, true_labels, predicted_labels


    def train_test_model(self, model, model_name, models_folder, epochs, classes):
        """
        This trains and tests a model.
        It plots and saves loss and accuracy over time as well as a confusion matrix for the model's predictions.
        There is also a checkpoint saving functionality which allows for pausing the training which saves the model's current state, current epoch and accumulated losses and accuriacies.

        Args:
            model (obj): A PyTorch model
            model_name (str): A given name for the model.
            models_folder (str): The folder path name containing the model.
            epochs (int): The number of epochs to train the model.
            classes (int): The classes in the dataset.
        """
        model, start_epoch, losses, accuracies = load_checkpoint(model, model_name, models_folder)
        
        try:
            for epoch in range(start_epoch, epochs):
                print(f'Epoch {epoch+1}\n-------------------------------')

                # Train the model and get the losses for this epoch
                train_losses = self.train(model)

                # Test the model and get the loss and accuracy for this epoch
                test_loss, true_labels, predicted_labels = self.test(model)
                losses.append(test_loss)
                accuracy = accuracy_score(true_labels, predicted_labels)
                accuracies.append(accuracy)  # Append the accuracy within the loop

                # Plotting the training loss and accuracy after each epoch
                print("Plotting training loss and accuracy")
                plt.figure()
                plt.plot(losses, label='Training Loss')
                plt.plot(accuracies, label='Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.title(f'Training Loss and Accuracy over Time ({epoch+1} Epochs)')
                plt.legend()
                plt.savefig(os.path.join(models_folder, f'{model_name}_loss_accuracy.png'))
                plt.close()
                
                print("Plotting confusion matrix")
                cm = confusion_matrix(true_labels, predicted_labels, normalize='all')
                plt.figure(figsize=(50, 50))
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.title(f'Confusion Matrix ({epoch+1} Epochs)')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(os.path.join(models_folder, f'{model_name}_confusion_matrix.png'))
                plt.close()

                # Save the checkpoint after each epoch
                save_checkpoint(model, model_name, models_folder, epoch+1, losses, accuracies)

        except KeyboardInterrupt:
            print('Training paused.')
            # Save the checkpoint when the training is paused
            save_checkpoint(model, model_name, models_folder, epoch, losses, accuracies)
        
        print('Training completed.')

# Save a checkpoint
def save_checkpoint(model, model_name, models_folder, end_epoch, losses, accuracies):
    """
    Creates a dictionary of checkpoint variables for a model.
    This inlcudes the model's state dictionary, the epoch in which it finished/stopped training, and the accumulated losses and accuracies from training.
    The checkpoint will be saved to disk as a PyTorch pt file.

    Args:
        model (obj): A PyTorch model.
        model_name (str): A given name for the model.
        models_folder (str): The folder path name containing the model.
        end_epoch (int): The epoch when training was finished/stopped.
        losses (obj): A list of losses per epoch from training.
        accuracies (obj): A list of accuracies per epoch from training.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'end_epoch': end_epoch,
        'losses': losses,
        'accuracies': accuracies
    }
    checkpoint_file = os.path.join(models_folder, f'{model_name}.pt')
    torch.save(checkpoint, checkpoint_file)
    print(f'Checkpoint saved: {checkpoint_file}\n')

# Load a checkpoint
def load_checkpoint(model, model_name, models_folder):
    """
    Loads a checkpoint for a model from a previous training.
    This will return the model with its previous state dictionary, the epoch to continue training from and the accumulated losses and accuracies from previous training.

    Args:
        model (obj): Number of images used for spikes per spike unit.
        model_name (int): Number of images used for noise.
        models_folder (str): The folder path name of the model.
        
    Returns:
        model (obj): A PyTorch model.
        start_epoch (int): The epoch to continue training from. 
        losses (obj): A list of losses per epoch from training.
        accuracies (obj): A list of accuracies per epoch from training.
    """
    checkpoint_file = os.path.join(models_folder, f'{model_name}.pt')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['end_epoch']
        losses = checkpoint['losses']
        accuracies = checkpoint['accuracies']
        print(f'Checkpoint loaded: {checkpoint_file}\nEnded training at epoch {start_epoch}\n')
        return model, start_epoch, losses, accuracies
    else:
        print('No checkpoint found.\nStart training at epoch 1\n')
        return model, 0, [], []

class TestModel():
    """
    A PyTorch-based class for testing a model.
    """
    def __init__(self, test_dataset, device, model):
        """
        Args:
            test_dataset (obj): The image dataset for testing.
            device (str): The name of the device to run training/testing.
            model (obj): A PyTorch model.
        """
        self.test_dataset = test_dataset
        self.device = device
        self.model = model
    
    def get_image_index_by_class(self, target_class):
        """
        Loads a checkpoint for a model from a previous training.

        Args:
            target_class (int): A specific class in the image dataset.

        Returns:
            int: The index of a random image belonging to the target class.
        """
        class_indices = [i for i, (_, label) in enumerate(self.test_dataset) if label == target_class]
        if class_indices:
            return random.choice(class_indices)  # Return the index of an image in the specified class
        else:
            return None  # Return None if no image of the specified class is found
    
    def get_confidence_probabilities(self, class_names, target_class):
        """
        This prints the model's confidence for the class an image actually belongs to as well as the class it believes the image belongs to.
        It also plots the model's confidence levels in all classes for that image.

        Args:
            class_names (obj): A list of class names in the image dataset.
            target_class (int): A specific class in the image dataset
        """
        self.model.eval()
        image_index = self.get_image_index_by_class(target_class)
        image, label = self.test_dataset[image_index]
        image = image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)  # Add batch dimension
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence = probabilities[label].item()  # Confidence of the true class
            predicted_class = torch.argmax(probabilities).item()  # Class with the highest probability
            predicted_confidence = probabilities[predicted_class].item()  # Confidence of the predicted class

        print(f"Confidence of True Class '{class_names[label]}': {confidence:.4f}")
        print(f"Confidence of Predicted Class: '{class_names[predicted_class]}': {predicted_confidence:.4f}")

        plt.figure(figsize=(20, 15))
        bars = plt.bar(class_names, probabilities.cpu())  # Move probabilities to CPU
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.xticks(rotation=90, fontsize=8)
        plt.ylim(0, 1.05) 

        # Add text annotations for probability values above the bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{prob:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.show()
    
class VisualizeModel():
    """
    A PyTorch-based class for visualizing different the filters of a model.
    """
    def __init__(self, model):
        """
        Args:
            model (obj): A PyTorch model.
        """
        self.model = model
        self.model_layers, self.model_weights = self.extract_layers_weights()
        
    def extract_layers_weights(self):
        """
        Extracts information about the convolutional layers in a model as well as their weights.

        Returns:
            model_layers (obj): A list of the convolutional layers in a model.
            model_weights (obj): A list of the weights of each layer in a model.
        """
        model_layers = [] # Save the conv layers in this list
        model_weights = [] # Save the conv layer weights in this list

        # Get all the model children as list
        model_children = list(self.model.children())

        # Append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv3d or type(model_children[i]) == nn.Conv2d:
                model_weights.append(model_children[i].weight)
                model_layers.append(model_children[i])

        return model_layers, model_weights

    def display_layers_weights(self):
        """
        Prints information about the convolutional layers in a model as well as their weights.
        """
        # Inspect the conv layers and the respective weights
        for i, (layer, weight) in enumerate(zip(self.model_layers, self.model_weights)):
            # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
            print(f"Layer {i}: {layer} ===> Shape: {weight.shape}")

    def visualize_layer_filters(self, layer_num, layer_shape):
        """
        This plots the filter for a convolutional layer in a model.
        The number of where the convolutional layer is located in the model has to be specified.
        This will be able to plot the filter for either a 3D or 2D convolutional layer.
        
        Args:
            layer_num (int): The number of a convolutional layer in a model.
            layer_shape (str): The shape of a convolutional layer in a model.
        """
        # Get the weights of the specified convolutional layer in the model
        model_layer = self.model_weights[layer_num].data

        # Check if the layer is 3D (e.g., for 3D convolutional layers)
        if layer_shape == '3D':
            # Extract the dimensions of the layer's weight tensor
            n_filters, in_channels, d, h, w = model_layer.shape

            # Set the number of columns for subplots to 4 (can be adjusted)
            ncols = 4

            # Calculate the number of rows required to display all filters
            nrows = int(np.ceil(n_filters / ncols))

            # Create a new figure with subplots and adjust spacing
            fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0.5)

            # Iterate through each subplot
            for i, ax in enumerate(axes.flat):
                ax.set_aspect('equal')
                ax.axis('off')
                # Check if there are more filters to display
                if i < n_filters:
                    # Get the data for the current filter and convert it to a NumPy array
                    filter_data = model_layer[i, 0].cpu().numpy()
                    # Display a central slice of the filter (you can adjust the slice as needed)
                    ax.imshow(filter_data[:, :, 1].T, cmap='gray')

        # Check if the layer is 2D (e.g., for 2D convolutional layers)
        elif layer_shape == '2D':
            # Extract the dimensions of the layer's weight tensor
            n_filters, in_channels, h, w = model_layer.shape

            # Set the number of columns for subplots to 8 (can be adjusted)
            ncols = 8

            # Calculate the number of rows required to display all filters
            nrows = int(np.ceil(n_filters / ncols))

            # Calculate the aspect ratio to maintain filter shape
            aspect_ratio = h / w

            # Create a new figure with subplots and adjust spacing
            fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0, wspace=0)

            # Iterate through each subplot
            for i, ax in enumerate(axes.flat):
                ax.set_aspect(aspect_ratio)
                ax.axis('off')
                # Check if there are more filters to display
                if i < n_filters:
                    # Get the data for the current filter and convert it to a NumPy array
                    filter_data = model_layer[i, 0].cpu().numpy()
                    # Display the filter using a grayscale colormap
                    ax.imshow(filter_data, cmap='gray')

        # Display the created subplots
        plt.show()
