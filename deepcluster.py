import numpy as np

import torch
from torch.utils.data import DataLoader

import time
from tqdm import tqdm

from sklearn.decomposition import PCA

import spikeinterface.full as si

import os
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import feature_extraction
from dataset import ClusteredDataset 
from training import TrainModel 


def train_deepcluster(dataset, dataloader, model, gmm, num_classes, kwargs_dict):
    """
    Trains a DeepCluster model using Gaussian Mixture Models (GMM) for clustering.

    It involves feature extraction, clustering using GMM, cluster matching, label reassignment,
    and classification. The process monitors cluster changes to identify convergence for early stopping.

    Parameters:
    - dataset: Dataset object containing image data.
    - dataloader: DataLoader object for batching dataset during feature extraction.
    - model: The neural network model used for feature extraction.
    - gmm: Gaussian Mixture Model used for clustering preprocessed features.
    - num_classes: The number of classes (clusters) to identify in the dataset.
    - kwargs_dict: Dictionary containing various keyword arguments, including:
        - 'epochs': Number of training epochs.
        - 'device': Computational device used (e.g., 'cuda', 'cpu').
        - 'device_ids': List of device IDs for multi-device training.
        - 'loss_fn': Loss function for the model training.
        - 'optimizer': Optimizer for model training.
        - 'cluster_matching': The method used for matching clusters. Options are 'sort_means' or 'hungarian'.

    Returns:
    - cluster_labels: Final clustering labels for the dataset images.
    """
    
    previous_cluster_labels = None  # To store cluster labels from the previous epoch
    
    convergence_epochs = 0  # Counter to track the number of epochs since last significant cluster change
    threshold_epochs = 5  # Number of epochs with minimal cluster change to consider convergence
    
    least_change = None
    threshold_change = len(dataset) * 0.005 # Threshold for determining significant cluster change
    cluster_changes = [0]  # Initialize a list to track changes in cluster labels

    image_paths = dataset.image_paths
    peak_times = get_peak_times(image_paths)
    sampling_frequency = kwargs_dict['sampling_frequency']
    
    start_time = time.time()  # Record the start time of the training process
    
    for epoch in range(kwargs_dict['epochs']):
        print(f"Epoch {epoch + 1}")

        # Step 1: Feature Extraction
        print("Feature Extraction")
        # Extract features using the neural network model
        features = feature_extraction.extract_features(dataloader, 500, model, kwargs_dict['device'], kwargs_dict['device_ids'])
        # Preprocess the features for clustering
        preprocessed_features = feature_extraction.preprocess_features(features, n_components=num_classes)

        # Step 2: Clustering
        print("Clustering")
        # Fit the GMM model to the preprocessed features
        gmm.fit(preprocessed_features)
        
        # Predict cluster labels for the features based on the fitted GMM
        current_cluster_labels = gmm.predict(preprocessed_features)
        # Convert labels to a NumPy array
        current_cluster_labels = current_cluster_labels.cpu().numpy() 
        
        # Cluster matching if not the first epoch
        if previous_cluster_labels is not None:
            cmp_start_time = time.time()
            
            print("Cluster Comparison\n")
            previous_sorting = create_numpy_sorting(peak_times, previous_cluster_labels, sampling_frequency)
            current_sorting = create_numpy_sorting(peak_times, current_cluster_labels, sampling_frequency)
            
            # Run the comparison
            cmp_sorting = si.compare_two_sorters(
                sorting1=previous_sorting,
                sorting2=current_sorting,
                sorting1_name='Previous',
                sorting2_name='Current',
            )
                        
            # Check the match events
            print("- Match Events:")
            print(cmp_sorting.match_event_count, "\n")

            
            # Check the agreement scores
            print("- Agreement Scores:")
            print(cmp_sorting.agreement_scores, "\n")
            
            cluster_change = get_cluster_change(cmp_sorting)
            cluster_changes.append(cluster_change)
            print(f"- Cluster Change: {cluster_change}")
            
            # Update convergence epochs based on the cluster change
            if least_change is None:
                least_change = cluster_change
            
            if cluster_change < least_change:
                least_change = cluster_change
                if cluster_change <= threshold_change:
                    convergence_epochs += 1
            
            print(f"- Least Change: {least_change}")
            print(f"- Convergence Epochs: [{convergence_epochs}/{threshold_epochs}]")
            
            cmp_end_time = time.time()
            cmp_time = cmp_end_time - cmp_start_time
            print(f"- Time: {cmp_time:.4f} seconds\n")
            
            if convergence_epochs == threshold_epochs:
                break                
         
        previous_cluster_labels = current_cluster_labels  # Update previous cluster labels for next iteration

        # Step 3: Label Reassignment
        # Create a new dataset and dataloader with the cluster labels
        clustered_dataset = ClusteredDataset(image_paths, current_cluster_labels)
        clustered_dataloader = DataLoader(clustered_dataset, batch_size=64)

        # Step 4: Classification
        # Train the model using the new dataloader with reassigned labels
        train_model = TrainModel(
            kwargs_dict['device'], kwargs_dict['device_ids'], 
            kwargs_dict['loss_fn'], kwargs_dict['optimizer'], 
            train_dataloader=clustered_dataloader
        )
        
        print("Classification")
        train_model.train(model, epoch, verbose_interval=50)
        print("\n")
       
    end_time = time.time()  # Record the end time of the training process
    total_time = end_time - start_time  # Calculate the total training time
    print(f"Total Time: {total_time:.4f} seconds")
    
    # Visualize the changes in cluster assignments and the final clustering in 3D
    plot_cluster_changes(cluster_changes, epoch)
    plot_clusters_3d(preprocessed_features, current_cluster_labels)
    
    return current_cluster_labels


def get_peak_times(image_paths):
    # Create an array to store peak times, initialized to zeros
    peak_times = np.zeros(len(image_paths), dtype=int) 

    # Iterate through the image paths
    for i, path in enumerate(image_paths):  
      # Get just the file name from the full path
      file_name = os.path.basename(path)  

      # Use regex to extract the time from the file name
      time = re.search(r'frame_(\d+)', file_name).group(1)

      # Store the time in the peak_times array
      peak_times[i] = time

    return peak_times


def sort_times_labels(peak_times, cluster_labels):        
    # Get sort order indices
    sort_indices = np.argsort(peak_times)

    # Use indices to sort arrays
    peak_times = peak_times[sort_indices]
    cluster_labels = cluster_labels[sort_indices]
    
    return peak_times, cluster_labels


def create_numpy_sorting(peak_times, cluster_labels, sampling_frequency):        
    peak_times, cluster_labels = sort_times_labels(peak_times, cluster_labels)
    
    numpy_sorting = si.NumpySorting.from_times_labels(peak_times, cluster_labels, sampling_frequency)
    
    return numpy_sorting


def get_cluster_change(cmp_sorting):
    match_event_count = cmp_sorting.match_event_count
    match_event_count = match_event_count.to_numpy()

    matched_units, _ = cmp_sorting.get_matching()
    matched_units = [int(unit) for unit in matched_units]

    for prev_unit, curr_unit in enumerate(matched_units):
        match_event_count[prev_unit][curr_unit] = 0
   
    # Summing all the numbers remaining in the modified array, ignoring NaN values
    cluster_change = np.sum(match_event_count)
    
    return cluster_change
    

def plot_cluster_changes(cluster_changes, epochs):
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, epochs + 2), cluster_changes, marker='o', color='r')
    
    plt.title('Cluster Change per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cluster Change')
    
    plt.xticks(range(1, epochs + 2), rotation=-90)
    plt.grid(True)
    plt.show()
    

def plot_clusters_2d(preprocessed_features, cluster_assignments):
    pca2 = PCA(n_components=2)
    features_2d = pca2.fit_transform(preprocessed_features)
    unique_labels = np.unique(cluster_assignments)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        plt.scatter(features_2d[cluster_assignments == label, 0], features_2d[cluster_assignments == label, 1], c=[colors[i]], label=label)

    plt.show()


def plot_clusters_3d(preprocessed_features, cluster_assignments):
    pca3 = PCA(n_components=3)
    features_3d = pca3.fit_transform(preprocessed_features)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=cluster_assignments, cmap='jet')

    plt.show()