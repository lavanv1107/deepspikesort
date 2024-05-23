import datetime
import logging
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from isosplit6 import isosplit6
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score

import matplotlib.pyplot as plt

sys.path.append("..")
from load_dataset import ClusteredDataset
from util import AverageMeter, calculate_elapsed_time, print_epoch_header


class DeepSpikeSortPipeline():
    def __init__(self, dataset_folder, dataset, cnn, loss_fn, optimizer, accelerator, cluster_model, output_folder, session_id, verbose_count=25):
        """
        Initializes the training and validation setup.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset.
        cnn : torch.nn.Module
            The convolutional neural network.
        loss_fn : torch.nn.CrossEntropyLoss
            The loss function.
        optimizer : torch.optim.Adam
            The optimizer.
        accelerator : Accelerator
            The Accelerator instance for handling distributed training.
        session_id : str
            An ID for the training session.
        verbose_count : int, optional
            Number of times to print detailed progress, by default 50.

        Attributes
        ----------    
        train_dataset : torch.utils.data.Subset
            The training dataset.
        val_dataset : torch.utils.data.Subset
            The validation dataset.
        progress_logger : logging.Logger
            The logger object for logging progress.
        """
        self.dataset_folder = dataset_folder
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=128)
        
        self.cnn = cnn
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        
        self.cluster_model = cluster_model
        
        self.output_folder = output_folder
        self.session_id = session_id
        self.progress_logger = self.setup_logger()

        self.verbose_count = verbose_count
        
        
    def extract_features(self, feature_dim=500):    
        self.cnn, self.dataloader = self.accelerator.prepare(self.cnn, self.dataloader)       
        
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        num_samples = len(self.dataloader.dataset)
        features = torch.zeros(num_samples, feature_dim, device=self.accelerator.device)  

        # Total number of batches and formatting for logs
        total_batches = len(self.dataloader)
        padding = len(str(total_batches))
        verbose_interval = max(1, total_batches // self.verbose_count)  
        
        self.cnn.eval()
        
        if self.accelerator.is_main_process:
            print("- Extraction")
            print('Extracting features...')
            
        extract_end = time.time()
        batch_end = time.time() 
        start_idx = 0
        
        with torch.no_grad():
            for batch, X in enumerate(self.dataloader):
                # Measure data loading time
                data_time_meter.update(calculate_elapsed_time(batch_end))
                
                inputs_dict = {'x': X, 'feature_extraction': True}
                pred = self.cnn(inputs_dict)

                end_idx = start_idx + pred.shape[0]
                features[start_idx:end_idx] = pred
                start_idx = end_idx

                # Update batch processing time
                batch_time_meter.update(calculate_elapsed_time(batch_end))
                batch_end = time.time() 

                # Format batch processing time for logs
                formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

                # Log validation progress at specified intervals
                if batch % verbose_interval == 0 and self.accelerator.is_main_process:
                    formatted_batch = f"{batch+1:0{padding}d}"
                    print(f'{formatted_time} - [{formatted_batch}/{total_batches}]\t'
                          f'Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                          f'Data: {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})')

        # Move final features tensor to CPU and convert to numpy
        features = features.cpu().numpy()
        
        preprocessed_features = self.preprocess_features(features)
        
        if self.accelerator.is_main_process:
            print(f'\nExtraction time: {calculate_elapsed_time(extract_end):.3f}\n')

        return preprocessed_features


    def preprocess_features(self, features):
        # Create a PCA object and fit the data
        pca = PCA(n_components=50)
        features_reduced = pca.fit_transform(features)

        # Whitening
        scaler = StandardScaler()
        features_whitened = scaler.fit_transform(features_reduced)

        # l2-Normalization
        features_normalized = normalize(features_whitened, norm='l2', axis=1)

        return features_normalized
    
    
    def cluster_features(self, features):
        if self.accelerator.is_main_process:
            print("- Clustering")            
            print("Clustering features...")
        
        cluster_end = time.time()
        
        # labels = self.cluster_model.fit_predict(features)
        labels = isosplit6(features)
        
        if self.accelerator.is_main_process:
            print(f'\nClustering time: {calculate_elapsed_time(cluster_end):.3f}\n')   
            print(len(set(labels)))
        
        return labels
        
    
    def train_cnn(self, labels):
        """
        Trains the convolutional neural network.

        Returns
        -------
        float
            The average loss over the training dataset.
        """
        clustered_dataset = ClusteredDataset(self.dataset_folder, self.dataset.trace_indices, labels)
        clustered_dataloader = DataLoader(clustered_dataset, batch_size=128)
            
        # Prepare the CNN, optimizer, and dataloader with the accelerator
        self.cnn, self.optimizer, clustered_dataloader = self.accelerator.prepare(
            self.cnn, self.optimizer, clustered_dataloader
        )

        # Initialize meters for tracking time and loss
        loss_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        # Total number of batches and formatting for logs
        total_batches = len(clustered_dataloader)
        padding = len(str(total_batches))
        verbose_interval = max(1, total_batches // self.verbose_count) 
        
        # Set the CNN to training mode
        self.cnn.train()
        
        if self.accelerator.is_main_process:
            print("- Training")
            print('Training CNN...')
            
        train_end = time.time()
        batch_end = time.time() 

        # Iterate over each batch
        for batch, (X, Y) in enumerate(clustered_dataloader):
            # Measure data loading time
            data_time_meter.update(calculate_elapsed_time(batch_end))

            # Zero out the gradients before forward pass
            self.optimizer.zero_grad()

            # Forward pass
            inputs_dict = {'x': X}
            pred = self.cnn(inputs_dict)

            # Compute loss
            Y = Y.long()
            loss = self.loss_fn(pred, Y)
            loss_meter.update(loss.item(), X.size(0))

            # Backward pass and update model parameters
            self.accelerator.backward(loss)
            self.optimizer.step()

            # Update batch processing time
            batch_time_meter.update(calculate_elapsed_time(batch_end))
            batch_end = time.time() 
            
            # Format batch processing time for logs
            formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

            # Periodically log training progress
            if batch % verbose_interval == 0 and self.accelerator.is_main_process:
                formatted_batch = f"{batch+1:0{padding}d}"
                print(f'{formatted_time} - [{formatted_batch}/{total_batches}]\t'
                      f'Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                      f'Data: {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})\t'
                      f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
                
        if self.accelerator.is_main_process:
            print(f'\nTraining time: {calculate_elapsed_time(train_end):.3f}')
                
        return loss_meter.avg
        
    
    def run_deepspikesort(self, num_epochs):
        total_end = time.time()  
        
        for epoch in range(1, num_epochs+1):
            try:
                if self.accelerator.is_main_process:
                    print_epoch_header(epoch)  

                # Step 1: Feature Extraction
                features = self.extract_features()

                # Step 2: Clustering
                labels = self.cluster_features(features)
                
                # Step 3: Training
                loss = self.train_cnn(labels)

                # Step 4: Evaluation
                silhouette = silhouette_score(features, labels)
                db_index = davies_bouldin_score(features, labels)
                ch_index = calinski_harabasz_score(features, labels)

                metrics = {'loss':loss, 'silhouette':silhouette, 'db_index':db_index, 'ch_index':ch_index}

                if self.accelerator.is_main_process:                    
                    # Log progress
                    self.log_progress(epoch, metrics) 
                
            except KeyboardInterrupt:
                break

        if self.accelerator.is_main_process:
            print(f"\nTotal time: {calculate_elapsed_time(total_end):.3f}") 

            np.save(os.path.join(self.output_folder, f'{self.session_id}_labels.npy'), labels)
            np.save(os.path.join(self.output_folder, f'{self.session_id}_times.npy'), self.dataset.times)
    
    
    def setup_logger(self):
        """
        Sets up a logger for recording progress.

        Parameters
        ----------
        progress_file : str
            The file path for logging progress.

        Returns
        -------
        logging.Logger
            The logger object for logging progress.
        """        
        file = os.path.join(self.output_folder, f'{self.session_id}_progress.log')   
        
        # Initialize a logger
        logger = logging.getLogger('progress_logger')
        logger.setLevel(logging.INFO)

        # Create a file handler to write logs to the file
        file_handler = logging.FileHandler(file, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # Set the formatter for the file handler
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)
        
        return logger


    def log_progress(self, epoch, metrics):       
        """
        Logs the progress of DeepSpikeSort.

        Parameters
        ----------        
        epoch : int
            The current epoch number.
        """
        self.progress_logger.info('[{0:03}]\t'
                                  'Loss: {1:.4f}\t'
                                  'Silhouette Score: {2:.4f}\t'
                                  'DB Index: {3:.4f}\t'
                                  'CH Index: {4:.4f}'
                                  .format(epoch, metrics['loss'], metrics['silhouette'], metrics['db_index'], metrics['ch_index']))