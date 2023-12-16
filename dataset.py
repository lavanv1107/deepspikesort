import os

import h5py

import numpy as np

import torch
from torch.utils.data import Dataset
                        
                        
class SupervisedDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and processing data from HDF5 files for supervised learning.

    Attributes:
        dataset_folder (str): Path to the dataset folder containing HDF5 files.
        unit_ids (list): List of unit IDs corresponding to HDF5 file names.
        shuffle (bool): Flag to shuffle the dataset.
        seed (int): Seed for reproducibility if shuffle is True.
        trace_indices (list): List of tuples, each containing unit ID and index within the file.
        trace_times (list): List containing times for each trace in the dataset.
    """

    def __init__(self, dataset_folder, unit_ids, shuffle=True, seed=0):
        """
        Initializes the SupervisedDataset instance.

        Args:
            dataset_folder (str): The folder path name of the dataset containing HDF5 files.
            unit_ids (list): A list containing unit IDs corresponding to HDF5 file names.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random seed for shuffling. Defaults to None.
        """
        self.dataset_folder = dataset_folder
        self.unit_ids = unit_ids
        self.shuffle = shuffle
        self.seed = seed
        
        # Initialize and gather data
        self.trace_indices, self.times = self.initialize_dataset()

    def initialize_dataset(self):
        """
        Initializes the dataset by loading trace indices and times from HDF5 files.

        This function iterates over a set of unit IDs, loads trace indices and corresponding times 
        from HDF5 files, and optionally shuffles them.

        If shuffling is enabled, it ensures that the trace indices and times are shuffled 
        together so that they still correspond to each other.

        Returns:
            trace_indices (list of (int, int)): List of tuples, each containing unit ID and index within the file.
            times (numpy.ndarray): Numpy array of trace times corresponding to each index.
        """
        trace_indices = []
        times = []  

        for unit_id in self.unit_ids:
            hdf5_file_path = os.path.join(self.dataset_folder, f"unit_{unit_id}.h5")
            with h5py.File(hdf5_file_path, 'r') as file:
                num_samples = file['traces'].shape[0]
                unit_indices = [(unit_id, i) for i in range(num_samples)]
                trace_indices.extend(unit_indices)
                times.extend(file['times'][:])

        # Convert trace_times_list to numpy array
        times = np.array(times, dtype='<i8')

        # Shuffle if required
        if self.shuffle:
            shuffle_indices = np.arange(len(trace_indices))
            np.random.seed(self.seed)
            np.random.shuffle(shuffle_indices)

            # Reorder trace_indices and trace_times using shuffled indices
            trace_indices = [trace_indices[i] for i in shuffle_indices]
            times = times[shuffle_indices]

        return trace_indices, times
    
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
    
    def get_labels(self):
        """
        Retrieves all labels corresponding to the traces in the dataset as a numpy array.

        Returns:
            numpy.ndarray: An array containing labels for each trace in the dataset.
        """
        labels = np.array([unit_id for unit_id, _ in self.trace_indices], dtype='<i8')
        return labels

    def __len__(self):
        """
        Returns the number of traces (and labels) in the dataset.

        Returns:
            int: The total number of traces in the dataset.
        """
        return len(self.trace_indices)

    def __getitem__(self, idx):
        """
        Retrieves a trace and its label from the dataset.

        Args:
            idx (int): The index of the trace and label pair.

        Returns:
            tuple: A tuple containing the trace tensor and its label.
        """
        unit_id, trace_idx = self.trace_indices[idx]
        trace = self.get_trace(unit_id, trace_idx)
        label = int(unit_id)

        return trace, label

    
class UnsupervisedDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and processing data from HDF5 files for unsupervised learning.

    Attributes:
        dataset_folder (str): Path to the dataset folder containing HDF5 files.
        unit_ids (list): List of unit IDs corresponding to HDF5 file names.
        shuffle (bool): Flag to shuffle the dataset.
        seed (int): Seed for reproducibility if shuffle is True.
        trace_indices (list): List of tuples, each containing unit ID and index within the file.
        trace_times (list): List containing times for each trace in the dataset.
    """

    def __init__(self, dataset_folder, unit_ids, shuffle=True, seed=0):
        """
        Initializes the UnsupervisedDataset instance.

        Args:
            dataset_folder (str): The folder path name of the dataset containing HDF5 files.
            unit_ids (list): A list containing unit IDs corresponding to HDF5 file names.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random seed for shuffling. Defaults to None.
        """
        self.dataset_folder = dataset_folder
        self.unit_ids = unit_ids
        self.shuffle = shuffle
        self.seed = seed
        
        # Initialize and gather data
        self.trace_indices, self.times = self.initialize_dataset()

    def initialize_dataset(self):
        """
        Initializes the dataset by loading trace indices and times from HDF5 files.

        This function iterates over a set of unit IDs, loads trace indices and corresponding times 
        from HDF5 files, and optionally shuffles them.

        If shuffling is enabled, it ensures that the trace indices and times are shuffled 
        together so that they still correspond to each other.

        Returns:
            trace_indices (list of (int, int)): List of tuples, each containing unit ID and index within the file.
            times (numpy.ndarray): Numpy array of trace times corresponding to each index.
        """
        trace_indices = []
        times = []  

        for unit_id in self.unit_ids:
            hdf5_file_path = os.path.join(self.dataset_folder, f"unit_{unit_id}.h5")
            with h5py.File(hdf5_file_path, 'r') as file:
                num_samples = file['traces'].shape[0]
                unit_indices = [(unit_id, i) for i in range(num_samples)]
                trace_indices.extend(unit_indices)
                times.extend(file['times'][:])

        # Convert trace_times_list to numpy array
        times = np.array(times, dtype='<i8')

        # Shuffle if required
        if self.shuffle:
            shuffle_indices = np.arange(len(trace_indices))
            np.random.seed(self.seed)
            np.random.shuffle(shuffle_indices)

            # Reorder trace_indices and trace_times using shuffled indices
            trace_indices = [trace_indices[i] for i in shuffle_indices]
            times = times[shuffle_indices]

        return trace_indices, times
    
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
    
    def get_labels(self):
        """
        Retrieves all labels corresponding to the traces in the dataset as a numpy array.

        Returns:
            numpy.ndarray: An array containing labels for each trace in the dataset.
        """
        labels = np.array([unit_id for unit_id, _ in self.trace_indices], dtype='<i8')
        return labels

    def __len__(self):
        """
        Returns the number of traces (and labels) in the dataset.

        Returns:
            int: The total number of traces in the dataset.
        """
        return len(self.trace_indices)

    def __getitem__(self, idx):
        """
        Retrieves a trace from the dataset.

        Args:
            idx (int): The index of the trace.

        Returns:
            torch.Tensor: The trace tensor.
        """
        unit_id, trace_idx = self.trace_indices[idx]
        trace = self.get_trace(unit_id, trace_idx)

        return trace
    
    
class ClusteredDataset(Dataset):
    """
    A custom PyTorch Dataset class for datasets where each image is assigned
    a cluster label instead of a traditional class label.

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
            

def select_units(peaks_matched, min_samples, max_samples, seed = 0, num_units = None):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Create a value_counts object of unit_index from peaks_matched
    units, samples = np.unique(peaks_matched['unit_index'], return_counts=True)

    # Filter units based on counts using boolean indexing
    filtered_units = units[(samples >= min_samples) & (samples <= max_samples)]

    if num_units is None or num_units >= len(filtered_units):
        selected_units = filtered_units
    else:
        selected_units = np.random.choice(filtered_units, size=num_units, replace=False)

    # Pad the selected_units with zeroes
    selected_units = [f"{unit:03d}" for unit in selected_units]

    return np.array(selected_units)