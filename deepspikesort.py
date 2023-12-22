import os

import time
import logging

import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spikeinterface.full as si

import feature_extraction 
from dataset import ClusteredDataset
from model import DeepCluster
from training import TrainModel 
import comparison
import util


def run_deepspikesort(dataset, num_units, num_classes, model, gmm, num_epochs, kwargs_dict):
    """
    It involves feature extraction, clustering using GMM, cluster comparison, label reassignment,
    and representation learning. The process monitors changes in cluster assignments between 
    epochs to identify convergence for early stopping.

    Parameters:
    - dataset:
    - num_classes: Number of classes (clusters) to identify in the dataset.
    - model: 
    - gmm: 
    - num_epochs: Number of epochs to run.
    - kwargs_dict: Dictionary containing various keyword arguments, including:
        - 'dataset_folder': Folder path name of the dataset.        
        - 'output_folder': Folder path name of the dataset.
        - 'results_folder': Folder path name of the dataset.
        - 'batch_size': Number of samples per batch for splitting the dataset.
        - 'sampling_frequency': Sampling frequency of recording.
        - 'device': Computational device used (e.g., 'cuda', 'cpu').
        - 'device_ids': List of device IDs for multi-device training.
        - 'loss_fn': Loss function for the model training.
        - 'optimizer': Optimizer for model training.
        - 'verbose_count': Number of times to print detailed progress.

    Returns:
    - cluster_labels: Final cluster labels for the dataset traces.
    """    
    trace_indices = dataset.trace_indices
    times = dataset.times
    
    dataloader = DataLoader(
        dataset,
        batch_size=kwargs_dict['batch_size']
    )    
    
    output_folder = kwargs_dict['output_folder']
    results_folder = kwargs_dict['results_folder']      
      
    metrics_progress_file = os.path.join(results_folder, f'metrics_progress_{num_units:0>3d}.log')        
    logger_dss = logging.getLogger('logger_dss')
    logger_dss.setLevel(logging.INFO)
    file_handler_dss = logging.FileHandler(metrics_progress_file, mode='w')
    formatter_dss = logging.Formatter('%(asctime)s - %(message)s')
    file_handler_dss.setFormatter(formatter_dss)
    logger_dss.addHandler(file_handler_dss)
    
    # Set metrics for convergence      
    previous_labels = None    
    aristabs = [0]
    threshold_aristab = 0.9
    
    convergence_counter = 0
    convergence_threshold = 10
    
    total_end = time.time()  
    for epoch in range(1, num_epochs+1):
        try:
            print(util.write_separator())
            print(f"Epoch {epoch}")  
            print(util.write_separator())

            # Step 1: Feature Extraction
            print("- Extraction")
            print('Extracting features...')        
            extract_end = time.time()

            # Extract features using the neural network model
            features = feature_extraction.extract_features(model, dataloader, 500, epoch, kwargs_dict['device'], kwargs_dict['device_ids'], kwargs_dict['verbose_count'])
            # Preprocess the features for clustering
            preprocessed_features = feature_extraction.preprocess_features(features, n_components=num_classes)

            print(f'\nExtraction time: {time.time() - extract_end:.3f}\n')

            # Step 2: Clustering
            print("- Clustering")        
            clustering_end = time.time()

            # Fit the GMM model to the preprocessed features
            gmm.fit(preprocessed_features)        
            # Predict cluster labels for the features based on the fitted GMM
            cluster_labels = gmm.predict(preprocessed_features)
            # Convert labels to a NumPy array
            cluster_labels = cluster_labels.cpu().numpy()

            print(f'\nClustering time: {time.time() - clustering_end:.3f}\n')

            # Step 3: Cluster Comparison    
            if previous_labels is not None:            
                print("- Comparison")
                print("Calculating adjusted rand score...")
                compare_end = time.time()

                aristab = adjusted_rand_score(previous_labels, cluster_labels)
                aristabs.append(aristab)    

                log_metric_progress(logger_dss, aristab, num_units, epoch, results_folder)      

                if aristab >= threshold_aristab:
                    convergence_counter += 1              
                else:
                    convergence_counter = 0
                    if epoch % 25 == 0:
                        threshold_aristab -= 0.1

                print(f'Convergence: [{convergence_counter}/{convergence_threshold}]')

                print(f'\nComparison time: {time.time() - compare_end:.3f}\n')

            if convergence_counter == convergence_threshold:
                    break

            previous_labels = cluster_labels  # Update previous cluster labels for next iteration

            # Step 4: Training
            # Create a new dataset and dataloader with the cluster labels
            clustered_dataset = ClusteredDataset(kwargs_dict['dataset_folder'], trace_indices, cluster_labels)
            clustered_dataloader = DataLoader(clustered_dataset, batch_size=kwargs_dict['batch_size'])

            # Train the model using the new dataloader with reassigned labels             
            train_model = TrainModel(
                kwargs_dict['device'], kwargs_dict['device_ids'], 
                kwargs_dict['loss_fn'], kwargs_dict['optimizer'], 
                train_dataloader=clustered_dataloader, verbose_count = kwargs_dict['verbose_count'],
            )

            print("- Training")
            print('Training model...')
            train_end = time.time()

            train_model.train(model, epoch)

            print(f'\nTraining time: {time.time() - train_end:.3f}')
                
        except KeyboardInterrupt:
            break
       
    print(f"\nTotal time: {time.time() - total_end:.3f}")    
    
    np.save(os.path.join(output_folder, f'features_{num_units:0>3d}.npy'), preprocessed_features)
    np.save(os.path.join(output_folder, f'dss_labels_{num_units:0>3d}.npy'), cluster_labels)
    np.save(os.path.join(output_folder, f'dss_times_{num_units:0>3d}.npy'), times)
    
    # Visualize the changes in cluster assignments   
    metrics = {
        'ARI Stability': {'values': aristabs, 'threshold': threshold_aristab},       
    }
    plot_metric_progress(metrics, num_units, epoch, results_folder)
    
    return cluster_labels, times
    
    
def log_metric_progress(logger_dss, aristab, num_units, epoch, results_folder):       
    logger_dss.info('Epoch: [{0}]\t'
                    'ARI Stability: {1:.4f}'
                    .format(epoch, aristab))
    
    
def plot_metric_progress(metrics, num_units, num_epochs, results_folder):
    # Calculate the width of the figure based on the number of epochs
    base_width = 10
    dynamic_width = (num_epochs / 100) * base_width

    # Ensure a minimum width is maintained
    dynamic_width = max(base_width, dynamic_width)

    # Number of subplots
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(dynamic_width, 5 * num_metrics))

    # Check if axs is a list (when there are multiple subplots)
    if num_metrics == 1:
        axs = [axs]

    # Plot each metric in a separate subplot
    for ax, (metric_type, metric_info) in zip(axs, metrics.items()):
        ax.plot(range(1, num_epochs + 1), metric_info['values'], marker='o', color='r')
        ax.hlines(metric_info['threshold'], 1, num_epochs, colors='blue', linestyles='--')
        ax.set_title(f'{metric_type}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_type)
        ax.grid(True)
        
        ax.set_xlim(1, num_epochs)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(results_folder, f"metrics_progress_{num_units:0>3d}.png"))
    plt.close(fig)