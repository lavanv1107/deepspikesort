import os
import glob

import numpy as np

import torch
from torch.utils.data import Dataset
import random


class SupervisedDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and processing image data. It converts 
    images from the dataset into tensors and attaches corresponding labels to them.
    
    Attributes:
        dataset_folder (str): Path to the dataset folder.
        unit_ids (list): List of unit IDs corresponding to folder names.
        shuffle (bool): Flag to shuffle the dataset.
        seed (int): Seed for reproducibility if shuffle is True.
        image_paths (list): List of paths to the images in the dataset.
        shuffled_indices (list): List of shuffled indices if shuffle is True.
    """

    def __init__(self, dataset_folder, unit_ids, shuffle=False, seed=None):
        """
        Initializes the SupervisedDataset instance.

        Args:
            dataset_folder (str): The folder path name of the image dataset.
            unit_ids (list): A list containing unit IDs corresponding to folder names in the image dataset.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random seed for shuffling. Defaults to None.
        """
        self.dataset_folder = dataset_folder
        self.unit_ids = unit_ids
        self.shuffle = shuffle
        self.seed = seed
        self.image_paths = self.get_image_paths()
        if self.shuffle:
            self.shuffled_indices = self.shuffle_indices()
            self.image_paths = self.get_shuffled_image_paths()

    def get_image_paths(self):
        """
        Generates a list of image paths from the dataset.

        Returns:
            list: A list containing paths to all images in the dataset.
        """
        image_paths = [path for folder in self.unit_ids 
                       for path in glob.glob(os.path.join(self.dataset_folder, str(folder), '*.npy'))]
        return image_paths
    
    def shuffle_indices(self):
        """
        Shuffles the indices of the dataset, using a set seed for reproducibility.

        Returns:
            list: A list of shuffled indices.
        """
        if self.seed is not None:
            random.seed(self.seed)
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        return indices
    
    def get_shuffled_image_paths(self):
        """
        Retrieves the list of image paths in the order they were shuffled.

        Returns:
            list: The shuffled image paths.
        """
        return [self.image_paths[i] for i in self.shuffled_indices]
    
    def get_image(self, idx):
        """
        Loads and returns an image as a tensor at a specified index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: The image converted to a tensor.
        """
        image = torch.from_numpy(np.load(self.image_paths[idx])).unsqueeze(0).float()
        return image
    
    def get_label(self, idx):
        """
        Retrieves the label (unit ID) corresponding to the image at the specified index.

        Args:
            idx (int): The index of the image.

        Returns:
            int: The label (unit ID) of the image.
        """
        label = int(os.path.dirname(self.image_paths[idx]).split(os.sep)[-1])
        return label
    
    def get_labels(self):
        """
        Retrieves labels for all images in the dataset.

        Returns:
            list: A list of labels corresponding to each image in the dataset.
        """
        labels = [int(os.path.dirname(path).split(os.sep)[-1]) for path in self.image_paths]
        return labels
    
    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label from the dataset.

        Args:
            idx (int): The index of the image and label pair.

        Returns:
            tuple: A tuple containing the image tensor and its label.
        """
        image = self.get_image(idx)
        label = self.get_label(idx)
        return image, label

    
class UnsupervisedDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and processing image data. It converts 
    images from the dataset into tensors.

    Attributes:
        dataset_folder (str): Path to the dataset folder.
        unit_ids (list): List of unit IDs corresponding to folder names.
        shuffle (bool): Flag to shuffle the dataset.
        seed (int): Seed for reproducibility if shuffle is True.
        image_paths (list): List of paths to the images in the dataset.
        shuffled_indices (list): List of shuffled indices if shuffle is True.
    """

    def __init__(self, dataset_folder, unit_ids, shuffle=False, seed=None):
        """
        Initializes the UnsupervisedDataset instance.

        Args:
            dataset_folder (str): The folder path name of the image dataset.
            unit_ids (list): A list containing unit IDs corresponding to folder names in the image dataset.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random seed for shuffling. Defaults to None.
        """
        self.dataset_folder = dataset_folder
        self.unit_ids = unit_ids
        self.shuffle = shuffle
        self.seed = seed
        self.image_paths = self.get_image_paths()
        if self.shuffle:
            self.shuffled_indices = self.shuffle_indices()
            self.image_paths = self.get_shuffled_image_paths()

    def get_image_paths(self):
        """
        Generates a list of image paths from the dataset.

        Returns:
            list: A list containing paths to all images in the dataset.
        """
        image_paths = [path for folder in self.unit_ids 
                       for path in glob.glob(os.path.join(self.dataset_folder, str(folder), '*.npy'))]
        return image_paths
    
    def shuffle_indices(self):
        """
        Shuffles the indices of the dataset, using a set seed for reproducibility.

        Returns:
            list: A list of shuffled indices.
        """
        if self.seed is not None:
            random.seed(self.seed)
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        return indices
    
    def get_shuffled_image_paths(self):
        """
        Retrieves the list of image paths in the order they were shuffled.

        Returns:
            list: The shuffled image paths.
        """
        return [self.image_paths[i] for i in self.shuffled_indices]
    
    def get_image(self, idx):
        """
        Loads and returns an image as a tensor at a specified index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: The image converted to a tensor.
        """
        image = torch.from_numpy(np.load(self.image_paths[idx])).unsqueeze(0).float()
        return image
    
    def get_labels(self):
        """
        Retrieves labels for all images in the dataset.

        Returns:
            list: A list of labels corresponding to each image in the dataset.
        """
        labels = [int(os.path.dirname(path).split(os.sep)[-1]) for path in self.image_paths]
        return labels

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset and converts it into a tensor.

        Args:
            idx (int): The index of the image.

        Returns:
            torch.Tensor: The image at the specified index, converted to a tensor.
        """
        image = self.get_image(idx)
        return image
    
    
class ClusteredDataset(Dataset):
    """
    A custom PyTorch Dataset class for datasets where each image is assigned
    a cluster label instead of a traditional class label.

    Attributes:
        image_paths (list): List of paths to the images in the dataset.
        cluster_labels (list): List of cluster labels corresponding to each image.
    """

    def __init__(self, image_paths, cluster_labels):
        """
        Initializes the ClusteredDataset instance.

        Args:
            image_paths (list): A list containing paths to all images in the dataset.
            cluster_labels (list): A list containing cluster labels for each image in the dataset.
        """
        self.image_paths = image_paths
        self.cluster_labels = cluster_labels
    
    def get_image(self, idx):
        """
        Loads and returns an image as a tensor at a specified index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: The image converted to a tensor.
        """
        image = torch.from_numpy(np.load(self.image_paths[idx])).unsqueeze(0).float()
        return image
    
    def get_label(self, idx):
        """
        Retrieves the cluster label of an image at a specified index.

        Args:
            idx (int): The index of the image.

        Returns:
            int: The cluster label of the image.
        """
        label = self.cluster_labels[idx]
        return label

    def __len__(self):
        """
        Returns the number of images (and labels) in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding cluster label from the dataset.

        Args:
            idx (int): The index of the image.

        Returns:
            tuple: A tuple containing the image tensor and its cluster label.
        """
        image = self.get_image(idx)
        label = self.get_label(idx)
        return image, label