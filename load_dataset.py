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
    A PyTorch Dataset class for loading trace data from HDF5 files for training.
    """

    def __init__(self, dataset_folder, shuffle=True, seed=0, channel_locations=None, method=None):
        """
        Initializes the TraceDataset instance for unsupervised training.

        Parameters
        ----------
        dataset_folder : str
            The folder path name containing the `peaks.h5` file with the data.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True.
        seed : int, optional
            Random seed for shuffling, by default 0.
        channel_locations : numpy.ndarray, optional
            Structured array containing channel IDs and their x, y locations on the probe, by default None.
        method : str, optional
            Method to apply to the traces ('mask', 'slice', etc.), by default None.
        """
        self.dataset_folder = dataset_folder
        
        self.shuffle = shuffle
        self.seed = seed
        
        self.channel_locations = channel_locations
        self.method = method
        
        # Prepare the dataset
        self.properties, self.trace_inds = self.prepare_dataset()

    def prepare_dataset(self):
        """
        Prepares the dataset by loading and organizing trace indices and properties.

        Returns
        -------
        properties : numpy.ndarray
            Array of properties for each trace.
        trace_inds : list or numpy.ndarray
            List or array of trace indices.
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

    def shuffle_dataset(self, properties, trace_inds):
        """
        Shuffles the dataset while maintaining the correlation between trace indices and properties.

        Parameters
        ----------
        properties : numpy.ndarray
            The array of properties corresponding to the trace indices.
        trace_inds : list or numpy.ndarray
            The list or array of trace indices to be shuffled.

        Returns
        -------
        properties : numpy.ndarray
            The shuffled array of properties.
        trace_inds : list or numpy.ndarray
            The shuffled list or array of trace indices.
        """
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        # Generate a shuffled array of indices
        inds_shuffled = np.random.permutation(len(trace_inds))

        # Shuffle properties and trace_inds using the generated indices
        properties = properties[inds_shuffled]
        trace_inds = [trace_inds[idx_shuffled] for idx_shuffled in inds_shuffled]
        
        return properties, trace_inds

    def get_trace(self, trace_idx):
        """
        Loads a trace from an HDF5 file at the specified index.

        Parameters
        ----------
        trace_idx : int
            The index of the trace in the HDF5 file.

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
        Retrieves a trace from the dataset.

        Parameters
        ----------
        idx : int
            The index of the trace.

        Returns
        -------
        torch.Tensor
            The trace tensor.
        """
        trace_idx = self.trace_inds[idx]
        trace = self.get_trace(trace_idx)
        
        # Apply the appropriate method to the trace
        if self.method == 'mask':
            trace = mask_trace(self.properties, idx, self.channel_locations, trace)
        elif self.method == 'slice':
            trace = slice_trace(self.properties, idx, self.channel_locations, trace)

        # Return just the trace for unsupervised learning
        return trace.unsqueeze(0).float()


class TraceDatasetEval(Dataset):
    """
    A PyTorch Dataset class for loading trace data from HDF5 files for evaluation.
    """

    def __init__(self, dataset_folder, dataset_type='unsupervised', unit_inds=None, 
                 num_samples='all', noise_samples=0, shuffle=True, seed=0, 
                 channel_locations=None, method=None):
        """
        Initializes the TraceDataset instance for evaluation.

        Parameters
        ----------
        dataset_folder : str
            The folder path name of the dataset containing HDF5 files.
        dataset_type : str, optional
            Type of the dataset to initialize ('supervised' or 'unsupervised'), by default 'unsupervised'.
        unit_inds : list, optional
            A list containing unit indices to include in the dataset.
        num_samples : int or str, optional
            Number of samples to select from each unit. If 'all', includes all samples, by default 'all'.
        noise_samples : int or str, optional
            Number of noise samples to include. If 'all', includes all noise samples, by default 0.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True.
        seed : int, optional
            Random seed for shuffling, by default 0.
        channel_locations : numpy.ndarray, optional
            Structured array containing channel IDs and their x, y locations on the probe, by default None.
        method : str, optional
            Method to apply to the traces ('mask', 'slice', etc.), by default None.
        """
        self.dataset_folder = dataset_folder
        self.dataset_type = dataset_type
        self.unit_inds = unit_inds
        self.num_samples = num_samples
        self.noise_samples = noise_samples
        
        self.shuffle = shuffle
        self.seed = seed
        
        self.channel_locations = channel_locations
        self.method = method
        
        # Prepare the dataset for evaluation
        self.properties, self.trace_inds = self.prepare_dataset()
        
        # Set up labels for supervised learning if needed
        if self.dataset_type == 'supervised':
            self.labels, self.labels_map = self.get_encoded_labels()
        else:
            self.labels, self.labels_map = None, None

    def prepare_dataset(self):
        """
        Prepares the dataset for evaluation by loading and organizing specific traces.

        Returns
        -------
        properties : numpy.ndarray
            Array of properties for each trace.
        trace_inds : numpy.ndarray
            Array of trace indices.
        """
        properties = []  
        trace_inds = []
        
        file = os.path.join(self.dataset_folder, f"peaks.h5")
        
        peaks_matched = np.load(os.path.join(self.dataset_folder, f"peaks_matched.npy"))
        
        with h5py.File(file, 'r') as handle:
            for unit_idx in self.unit_inds:
                # Get mask for this unit's peaks
                unit_mask = peaks_matched['unit_index'] == unit_idx
                
                # Get the original indices of these rows in the full peaks_matched array
                # This replaces the need for a 'peak_index' column
                original_indices = np.where(unit_mask)[0]
                
                # Determine the number of samples to load
                if unit_idx == -1:
                    num_samples = self.noise_samples if self.noise_samples != 'all' else len(original_indices)
                else:
                    num_samples = self.num_samples if self.num_samples != 'all' else len(original_indices)
                
                # Limit to the requested number of samples
                selected_indices = original_indices[:num_samples]
                trace_inds.extend(selected_indices)
                
                for i in selected_indices:
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
        Shuffles the dataset while maintaining the correlation between trace indices and properties.

        Parameters
        ----------
        properties : numpy.ndarray
            The array of properties corresponding to the trace indices.
        trace_inds : list or numpy.ndarray
            The list or array of trace indices to be shuffled.

        Returns
        -------
        properties : numpy.ndarray
            The shuffled array of properties.
        trace_inds : list or numpy.ndarray
            The shuffled list or array of trace indices.
        """
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        # Generate a shuffled array of indices
        inds_shuffled = np.random.permutation(len(trace_inds))

        # Shuffle properties and trace_inds using the generated indices
        properties = properties[inds_shuffled]
        trace_inds = [trace_inds[idx_shuffled] for idx_shuffled in inds_shuffled]
        
        return properties, trace_inds

    def get_trace(self, trace_idx):
        """
        Loads a trace from an HDF5 file at the specified index.

        Parameters
        ----------
        trace_idx : int
            The index of the trace in the HDF5 file.

        Returns
        -------
        torch.Tensor
            The trace data as a tensor.
        """
        file = os.path.join(self.dataset_folder, f"peaks.h5")        
        with h5py.File(file, 'r') as handle:           
            trace = torch.from_numpy(handle['traces'][trace_idx])
            
        return trace

    def get_labels(self):
        """
        Retrieves all labels corresponding to the traces in the dataset as a numpy array.
        Labels are extracted from the peaks_matched array.

        Returns
        -------
        numpy.ndarray
            An array containing labels for each trace in the dataset.
        """
        # Load the peaks_matched array and extract labels
        peaks_matched = np.load(os.path.join(self.dataset_folder, f"peaks_matched.npy"))
        labels = np.array([peaks_matched['unit_index'][i] for i in self.trace_inds], dtype='<i8')
        return labels
    
    def get_encoded_labels(self):
        """
        Transforms and retrieves encoded labels corresponding to the traces 
        in the dataset as a NumPy array.

        The labels are transformed to a contiguous range starting from 0.

        Returns
        -------
        tuple of numpy.ndarray and dict
            encoded_labels : numpy.ndarray
                An array containing the label-encoded labels for each trace in the dataset.
            labels_map : dict
                Mapping of encoded labels to original labels.
        """
        labels = self.get_labels()
        
        # Import here to avoid potential circular imports
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()        
        labels_encoded = np.array(label_encoder.fit_transform(labels), dtype='<i8')
        
        labels_map = dict(zip(labels_encoded, labels))
        
        return labels_encoded, labels_map
        
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
            trace = mask_trace(self.properties, idx, self.channel_locations, trace)
        elif self.method == 'slice':
            trace = slice_trace(self.properties, idx, self.channel_locations, trace)

        # Return the trace along with the label if supervised
        if self.dataset_type == 'supervised':
            label = self.labels[idx]
            return trace.unsqueeze(0).float(), label
        else:
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

    def __init__(self, dataset_folder, trace_inds, cluster_labels, properties=None, channel_locations=None, method=None):
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
        self.channel_locations = channel_locations
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
            trace = mask_trace(self.properties, idx, self.channel_locations, trace)
            
        label = self.cluster_labels[idx]
        
        return trace.unsqueeze(0).float(), label
    
    
def mask_trace(properties, idx, channel_locations, trace):
    channel_idx = properties[idx]['channel_index']       

    neighbor_channels = get_channel_neighbors(channel_locations, channel_idx, 80)['channel_index']

    channels_unmasked = np.append(neighbor_channels, channel_idx)    
    
    trace_masked = torch.zeros_like(trace)

    for channel in channels_unmasked:
        # Determine the channel's location (row and column)
        row, column = get_channel_ind_reshaped(channel) 

        # Apply masking for the specific channel
        trace_masked[:, row, column] = trace[:, row, column]

    return trace_masked


def slice_trace(times, idx, data, channel_locations, trace):
    time = times[idx]
    channel_ind = data['channel_index'][data['time'] == time][0]    

    channel_neighbors = get_channel_neighbors(channel_locations, channel_ind, 220)['channel_index']
    slice_min = get_channel_ind_reshaped(min(channel_neighbors))[0]
    slice_max = get_channel_ind_reshaped(max(channel_neighbors))[0]

    trace_slice = trace[:, slice_min:slice_max+1, :]
    
    return trace_slice
