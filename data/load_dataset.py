import os
import sys

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder
                        

def select_units(data, min_samples=0, max_samples='max', num_units='all', seed=0):
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
    filtered_units = units[(samples >= min_samples) & (samples <= max_samples)]
    
    # Determine the number of units to select
    num_units = len(filtered_units) if num_units == 'all' else num_units

    # Check if the number of units requested is available after filtering
    if num_units > len(filtered_units):
        print(f"Error: Required {num_units} units, but only found {len(filtered_units)} units within the specified range.")
        sys.exit()

    # Select units
    selected_units = np.random.choice(filtered_units, size=num_units, replace=False) if num_units < len(filtered_units) else filtered_units

    # Append a noise unit represented by -1
    # selected_units = np.append(selected_units, -1)
    
    return selected_units


class TraceDataset(Dataset):
    """
    A PyTorch Dataset class for loading trace data from HDF5 files. 
    """

    def __init__(self, dataset_folder, dataset_type, unit_ids, num_samples='all', noise_samples=0, shuffle=True, seed=0):
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
        
        self.dataset_type = dataset_type
        
        self.unit_ids = unit_ids
        self.num_samples = num_samples
        self.noise_samples = noise_samples
        
        self.shuffle = shuffle
        self.seed = seed
        
        # Prepares the dataset
        self.trace_indices, self.times = self.prepare_dataset()
        
        if self.dataset_type == 'supervised':
            self.labels, self.labels_map = self.get_encoded_labels()
        else:
            self.labels, self.labels_map = None, None

    def prepare_dataset(self):
        """
        Prepares the dataset by loading and organizing trace indices and times.

        Returns
        -------
        tuple of list and numpy.ndarray
            trace_indices : list of (int, int)
                List of tuples, each containing unit ID and index within the file.
            times : numpy.ndarray
                Numpy array of trace times corresponding to each index.
        """
        # List to hold the indices of each trace and their corresponding times
        trace_indices = []
        times = []
        
        # Pad unit indices with zeroes and convert to strings 
        unit_ids = [f"{unit_id:03}" for unit_id in self.unit_ids]

        # Loop through each unit and load data from its corresponding HDF5 file
        for unit_id in unit_ids:
            hdf5_file_path = os.path.join(self.dataset_folder, f"unit_{unit_id}.h5")
            
            try:
                # Open the HDF5 file and extract relevant data
                with h5py.File(hdf5_file_path, 'r') as file:
                    # Determine the number of samples to load
                    if unit_id == '-01':
                        num_samples = self.noise_samples if self.noise_samples != 'all' else file['traces'].shape[0]
                    else:
                        num_samples = self.num_samples if self.num_samples != 'all' else file['traces'].shape[0]

                    # Create index pairs for each sample in this unit
                    unit_indices = [(unit_id, i) for i in range(num_samples)]
                    trace_indices.extend(unit_indices)

                    # Add corresponding times for each sample
                    times.extend(file['times'][:])
            except OSError as e:
                print(f"Error opening file: {hdf5_file_path}. Exception: {e}")
                continue

        # Convert the list of times to a numpy array
        times = np.array(times, dtype='<i8')

        # Shuffle the dataset
        if self.shuffle:
            trace_indices, times = self.shuffle_dataset(trace_indices, times)

        return trace_indices, times

    def shuffle_dataset(self, trace_indices, times):
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
        shuffle_indices = np.random.permutation(len(trace_indices))

        # Shuffle trace_indices and times using the generated indices
        trace_indices = [trace_indices[i] for i in shuffle_indices]
        times = times[shuffle_indices]
        
        return trace_indices, times

    def get_trace(self, unit_id, idx):
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
        hdf5_file_path = os.path.join(self.dataset_folder, f"unit_{unit_id}.h5")
        with h5py.File(hdf5_file_path, 'r') as file:
            trace = torch.from_numpy(file['traces'][idx]).unsqueeze(0).float()
        return trace
    
    def get_labels(self):
        """
        Retrieves all labels corresponding to the traces in the dataset as a numpy array.

        Returns
        -------
        numpy.ndarray
            An array containing labels for each trace in the dataset.
        """
        labels = np.array([unit_id for unit_id, _ in self.trace_indices], dtype='<i8')
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
        label_encoder = LabelEncoder()
        encoded_labels = np.array(label_encoder.fit_transform(labels), dtype='<i8')
        labels_map = dict(zip(encoded_labels, labels))
        
        return encoded_labels, labels_map

    def __len__(self):
        """
        Returns the number of traces in the dataset.

        Returns
        -------
        int
            The total number of traces in the dataset.
        """
        return len(self.trace_indices)

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
        unit_id, trace_idx = self.trace_indices[idx]
        trace = self.get_trace(unit_id, trace_idx)

        if self.dataset_type == 'supervised':
            label = self.labels[idx]
            return trace, label
        elif self.dataset_type == 'unsupervised':
            return trace

    
class ClusteredDataset(Dataset):
    """
    A custom PyTorch Dataset class for datasets where each image is assigned
    a cluster label.

    Attributes:
        dataset_folder (str): The folder path name of the dataset containing HDF5 files.
        trace_indices (list): List of tuples, each containing unit ID and index within the file.
        cluster_labels (list): List of cluster labels corresponding to each image.
    """

    def __init__(self, dataset_folder, trace_indices, cluster_labels):
        """
        Initializes the ClusteredDataset instance.

        Args:
            dataset_folder (str): The folder path name of the dataset containing HDF5 files.
            trace_indices (list): List of tuples, each containing unit ID and index within the file.
            cluster_labels (list): A list containing cluster labels for each image in the dataset.
        """
        self.dataset_folder = dataset_folder
        self.trace_indices = trace_indices
        self.cluster_labels = cluster_labels
    
    def get_trace(self, unit_id, idx):
        """
        Loads a trace from an HDF5 file at the specified index.

        Args:
            unit_id (int): Unit ID corresponding to the HDF5 file.
            trace_idx (int): The index of the trace in the unit's file.

        Returns:
            torch.Tensor: The trace data as a tensor.
        """
        hdf5_file_path = os.path.join(self.dataset_folder, f"unit_{unit_id}.h5")
        with h5py.File(hdf5_file_path, 'r') as file:
            trace = torch.from_numpy(file['traces'][idx]).unsqueeze(0).float()
        return trace

    def __len__(self):
        """
        Returns the number of traces (and labels) in the dataset.

        Returns:
            int: The total number of traces in the dataset.
        """
        return len(self.trace_indices)

    def __getitem__(self, idx):
        """
        Retrieves a trace and its corresponding cluster label from the dataset.

        Args:
            idx (int): The index of the trace.

        Returns:
            tuple: A tuple containing the trace tensor and its cluster label.
        """
        unit_id, trace_idx = self.trace_indices[idx]
        trace = self.get_trace(unit_id, trace_idx)
        label = self.cluster_labels[idx]
        
        return trace, label
            
