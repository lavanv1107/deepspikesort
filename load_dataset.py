import os
import sys
import random

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

sys.path.append("..")
from preprocessing import get_channel_neighbors, get_channel_ind_reshaped
                        

def select_units(data, num_units='all', min_samples=0, max_samples='max', seed=0):
    """
    Selects unit indices from data within a specified range of sample counts, including a noise unit.

    Parameters
    ----------
    data : numpy.ndarray
        The data containing 'unit_index'.
    min_samples : int, optional
        Minimum number of samples for a unit to be considered, by default 0.
    max_samples : int or str, optional
        Maximum number of samples for a unit to be considered. If 'max', uses the maximum available, by default 'max'.
    num_units : int or str, optional
        Number of units to select randomly. If 'all', selects all units within the range, by default 'all'.
    seed : int, optional
        Seed for random number generator for reproducibility, by default 0.

    Returns
    -------
    numpy.ndarray
        Array of selected unit indices with an additional noise unit represented by -1.
    """
    # Set the seed for reproducibility
    np.random.seed(seed)
  
    # Extract unique unit indices and their corresponding sample counts, excluding the noise unit
    units, samples = np.unique(data[data['unit_index'] != -1]['unit_index'], return_counts=True)
    
    # Determine the maximum number of samples, either as specified or the largest sample count available
    max_samples = samples.max() if max_samples == 'max' else max_samples
    
    # Filter units based on the specified min and max sample counts
    units_filtered = units[(samples >= min_samples) & (samples <= max_samples)]
    
    # Determine the number of units to select
    num_units = len(units_filtered) if num_units == 'all' else num_units

    # Check if the number of units requested is available after filtering
    if num_units > len(units_filtered):
        print(f"Error: Required {num_units} units, but only found {len(units_filtered)} units within the specified range.")
        sys.exit()

    # Select units
    units_selected = np.random.choice(units_filtered, size=num_units, replace=False) if num_units < len(units_filtered) else units_filtered
    
    return units_selected


class TraceDataset(Dataset):
    """
    A PyTorch Dataset class for loading trace data from HDF5 files. 
    """

    def __init__(self, dataset_folder, mode, unit_inds=None, num_samples='all', noise_samples=0, shuffle=True, seed=0, channels=None, method=None):
        """
        Initializes the TraceDataset instance based on the specified dataset type 
        (supervised or unsupervised).

        Parameters
        ----------
        dataset_folder : str
            The folder path name of the dataset containing HDF5 files.
        dataset_type : str
            Type of the dataset to initialize ('supervised' or 'unsupervised').
        unit_ids : list
            A list containing unit IDs corresponding to HDF5 file names.
        num_samples : int or str, optional
            Number of samples to select from each unit. If 'all', includes all samples, by default 'all'.
        noise_samples : int or str, optional
            Number of noise samples to include. If 'all', includes all noise samples, by default 0.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True.
        seed : int, optional
            Random seed for shuffling, by default 0.
        data: numpy.ndarray
            A numpy array of detected peak events.

        Attributes
        ----------
        trace_indices : list of tuples
            List of tuples, each containing unit ID and index within the file.
        times : numpy.ndarray
            Numpy array of trace times corresponding to each index.
        labels : numpy.ndarray
            Array of labels for each trace in the dataset. Only relevant for supervised mode.
        labels_map : dict
            Mapping of encoded labels to original labels. Only relevant for supervised mode.
        """
        self.dataset_folder = dataset_folder
        self.mode = mode

        self.unit_inds = unit_inds
        self.num_samples = num_samples
        self.noise_samples = noise_samples
        
        self.shuffle = shuffle
        self.seed = seed
        
        self.channels = channels
        self.method = method
        
        # Prepares the dataset
        if mode == 'eval':
            self.properties, self.trace_inds = self.prepare_dataset_eval()
        else:
            self.properties, self.trace_inds = self.prepare_dataset()

    def prepare_dataset(self):
        """
        Prepares the dataset by loading and organizing trace indices and times.

        Returns
        -------
        trace_indices : list of (int, int)
            List of tuples, each containing unit ID and index within the file.
        times : numpy.ndarray
            Numpy array of trace times corresponding to each index.
        """
        properties = []
        trace_inds = []
        
        file = os.path.join(self.dataset_folder, f"peaks.h5")
        
        with h5py.File(file, 'r') as handle:
            trace_inds = range(handle['traces'].shape[0])
            properties = handle['properties'][:]

        # Shuffle the dataset
        if self.shuffle:
            properties, trace_inds = self.shuffle_dataset(properties, trace_inds)

        return properties[:100], trace_inds[:100]

    def prepare_dataset_eval(self):
        properties = []  
        trace_inds = []
        
        file = os.path.join(self.dataset_folder, f"peaks.h5")
        
        peaks_matched = np.load(os.path.join(self.dataset_folder, f"peaks_matched.npy"))
        with h5py.File(file, 'r') as handle:
            for unit_idx in self.unit_inds:
                peaks_matched_unit = peaks_matched[peaks_matched['unit_index']==unit_idx]
                # Determine the number of samples to load
                if unit_idx == -1:
                    num_samples = self.noise_samples if self.noise_samples != 'all' else peaks_matched_unit.shape[0]
                else:
                    num_samples = self.num_samples if self.num_samples != 'all' else peaks_matched_unit.shape[0]
                # Create index pairs for each sample in this unit
                unit_trace_inds = peaks_matched_unit['peak_index'][:num_samples]
                trace_inds.extend(unit_trace_inds)
                
                for i in unit_trace_inds:
                    properties.append(handle['properties'][i])
        
        # Convert to numpy array for easier shuffling (if needed)
        properties = np.array(properties)
        trace_inds = np.array(trace_inds)
        
        # Shuffle the dataset
        if self.shuffle:
            properties, trace_inds = self.shuffle_dataset(properties, trace_inds)
        return properties, trace_inds
        

    def shuffle_dataset(self, properties, trace_inds):
        """
        Shuffles the dataset while maintaining the correlation between trace indices and times.

        Parameters
        ----------
        trace_indices : list of tuples
            The list of trace indices to be shuffled.
        times : numpy.ndarray
            The array of trace times corresponding to the trace indices.

        Returns
        -------
        trace_indices : list of tuples
            The shuffled list of trace indices.
        times : numpy.ndarray
            The shuffled array of trace times.
        """
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        # Generate a shuffled array of indices
        inds_shuffled = np.random.permutation(len(trace_inds))

        # Shuffle trace_indices and times using the generated indices
        properties = properties[inds_shuffled]
        trace_inds = [trace_inds[idx_shuffled] for idx_shuffled in inds_shuffled]
        
        return properties, trace_inds

    def get_trace(self, trace_idx):
        """
        Loads a trace from an HDF5 file at the specified index.

        Parameters
        ----------
        unit_id : int
            Unit ID corresponding to the HDF5 file.
        idx : int
            The index of the trace in the unit's file.

        Returns
        -------
        torch.Tensor
            The trace data as a tensor.
        """
        file = os.path.join(self.dataset_folder, f"peaks.h5")        
        with h5py.File(file, 'r') as handle:           
            trace = torch.from_numpy(handle['traces'][trace_idx])
            
        return trace  
        
    def __len__(self):
        return len(self.properties)

    def __getitem__(self, idx):
        """
        Retrieves a trace from the dataset. If the dataset is supervised, it also 
        retrieves the corresponding label.

        Parameters
        ----------
        idx : int
            The index of the trace (and label, if in supervised mode).

        Returns
        -------
        If supervised:
            tuple of torch.Tensor and int
                trace : torch.Tensor
                    The trace tensor.
                label : int
                    The label of the trace.
        If unsupervised:
            torch.Tensor
                The trace tensor without a corresponding label.
        """
        trace_idx = self.trace_inds[idx]
        trace = self.get_trace(trace_idx)
        
        # Apply the appropriate method to the trace
        if self.method == 'mask':
            trace = mask_trace(self.properties, idx, self.channels, trace)
        elif self.method == 'slice':
            trace = slice_trace(self.times, idx, self.data, self.channels, trace)

        return trace.unsqueeze(0).float()

    
class ClusteredDataset(Dataset):
    """
    A custom PyTorch Dataset class for datasets where each image is assigned
    a cluster label.

    Attributes:
        dataset_folder (str): The folder path name of the dataset containing HDF5 files.
        trace_indices (list): List of tuples, each containing unit ID and index within the file.
        cluster_labels (list): List of cluster labels corresponding to each image.
    """

    def __init__(self, dataset_folder, trace_inds, cluster_labels, properties=None, channels=None, method=None):
        """
        Initializes the ClusteredDataset instance.

        Args:
            dataset_folder (str): The folder path name of the dataset containing HDF5 files.
            trace_indices (list): List of tuples, each containing unit ID and index within the file.
            cluster_labels (list): A list containing cluster labels for each image in the dataset.
        """
        self.dataset_folder = dataset_folder
        
        self.trace_inds = trace_inds
        self.cluster_labels = cluster_labels
        
        self.properties = properties
        self.channels = channels
        self.method = method
    
    def get_trace(self, trace_idx):
        """
        Loads a trace from an HDF5 file at the specified index.

        Args:
            unit_id (int): Unit ID corresponding to the HDF5 file.
            trace_idx (int): The index of the trace in the unit's file.

        Returns:
            torch.Tensor: The trace data as a tensor.
        """
        file = os.path.join(self.dataset_folder, f"peaks.h5")        
        with h5py.File(file, 'r') as handle:
            trace = torch.from_numpy(handle['traces'][trace_idx])
            
        return trace

    def __len__(self):
        return len(self.properties)

    def __getitem__(self, idx):
        """
        Retrieves a trace and its corresponding cluster label from the dataset.

        Args:
            idx (int): The index of the trace.

        Returns:
            tuple: A tuple containing the trace tensor and its cluster label.
        """
        trace_idx = self.trace_inds[idx]
        trace = self.get_trace(trace_idx)
        
        if self.method == 'mask':
            trace = mask_trace(self.properties, idx, self.channels, trace)
            
        label = self.cluster_labels[idx]
        
        return trace.unsqueeze(0).float(), label
    
    
def mask_trace(properties, idx, channels, trace):
    channel_idx = properties[idx]['channel_index']       

    neighbor_channels = get_channel_neighbors(channels, channel_idx, 80)['channel_index']

    channels_unmasked = np.append(neighbor_channels, channel_idx)    
    
    trace_masked = torch.zeros_like(trace)

    for channel in channels_unmasked:
        # Determine the channel's location (row and column)
        row, column = get_channel_ind_reshaped(channel) 

        # Apply masking for the specific channel
        trace_masked[:, row, column] = trace[:, row, column]

    return trace_masked


def slice_trace(times, idx, data, channels, trace):
    time = times[idx]
    channel_ind = data['channel_index'][data['time'] == time][0]    

    channel_neighbors = get_channel_neighbors(channels, channel_ind, 220)['channel_index']
    slice_min = get_channel_ind_reshaped(min(channel_neighbors))[0]
    slice_max = get_channel_ind_reshaped(max(channel_neighbors))[0]

    trace_slice = trace[:, slice_min:slice_max+1, :]
    
    return trace_slice