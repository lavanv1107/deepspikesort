import os
import psutil

import numpy as np
import multiprocessing as mp
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset
import random

import preprocessing


def display_resources():
    nthreads = psutil.cpu_count(logical=True)
    ncores = psutil.cpu_count(logical=False)
    nthreads_per_core = nthreads // ncores
    nthreads_available = len(os.sched_getaffinity(0))
    ncores_available = nthreads_available // nthreads_per_core

    assert nthreads == os.cpu_count()
    assert nthreads == mp.cpu_count()

    print(f'{nthreads=}')
    print(f'{ncores=}')
    print(f'{nthreads_per_core=}')
    print(f'{nthreads_available=}')
    print(f'{ncores_available=}')


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
    def __init__(self, dataset_folder, folder_labels):
        """
        Args:
            dataset_folder (str): The folder path name of the image dataset.
            folder_labels (list): A list containing folder names of image dataset.
        """
        self.dataset_folder = dataset_folder
        self.folder_labels = folder_labels
        self.image_paths = self.get_image_paths()
        random.shuffle(self.image_paths)

    def get_image_paths(self):
        """
        Creates a list of path names of all images in the image dataset.

        Returns:
            obj: A list of image path names.
        """
        image_paths = []
        # Iterate over the subfolders
        folder_labels = [str(x) for x in self.folder_labels]
        for folder in folder_labels:
            folder_path = os.path.join(self.dataset_folder, folder)
            folder_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
            image_paths.extend([os.path.join(folder_path, file) for file in folder_files])
        return image_paths
    
    def get_label(self, idx):
        """
        Retrieves the folder name corresponding to the image at a given index.

        Args:
            idx (int): Index number of image in the image dataset.

        Returns:
            folder_name (str): The folder name of the image.
        """
        label = os.path.dirname(self.image_paths[idx]).split(os.sep)[-1]
        label = int(label)
        return label
    
    def get_image(self, idx):
        image = torch.from_numpy(np.load(self.image_paths[idx])).unsqueeze(0).float()
        return image

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
        image = self.get_image(idx)
        # Extract the label from the folder name
        label = self.get_label(idx)
        return image, label

    
class ClusteredDataset(Dataset):
    def __init__(self, dataset_folder, folder_labels, folder_to_cluster_map):
        self.dataset_folder = dataset_folder
        self.folder_labels = folder_labels
        self.folder_to_cluster_map = folder_to_cluster_map  # The mapping from folder to cluster
        self.image_paths = self.get_image_paths()
        random.shuffle(self.image_paths)

    def get_image_paths(self):
        image_paths = []
        for folder in self.folder_labels:
            folder_path = os.path.join(self.dataset_folder, str(folder))
            folder_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
            image_paths.extend([os.path.join(folder_path, file) for file in folder_files])
        return image_paths
    
    def get_label(self, idx):
        folder_name = os.path.dirname(self.image_paths[idx]).split(os.sep)[-1]
        folder_name = int(folder_name)
        
        # Use the mapping to fetch the cluster assignment instead of folder_name
        label = self.folder_to_cluster_map.get(folder_name, folder_name)
        
        return label
    
    def get_image(self, idx):
        image = torch.from_numpy(np.load(self.image_paths[idx])).unsqueeze(0).float()
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the numpy file as a grayscale image tensor
        image = self.get_image(idx)
        # Extract the label from the folder name
        label = self.get_label(idx)
        
        return image, label