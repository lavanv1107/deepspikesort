import os
import sys

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.nn as nn

import warnings
warnings.simplefilter("ignore")

import dataset
from model import DeepCluster
from training import TrainModel 
import util


if __name__ == "__main__":        
    recording_num = int(sys.argv[1])
    results_folder = f'results/{recording_num:03}/classifier'
    os.makedirs(results_folder, exist_ok=True)
    
    # Load matched peaks object
    spikes_folder = f'spikes/{recording_num:03}'
    spikes_file = os.path.join(spikes_folder, "spikes_nwb.npy")
    spikes = np.load(spikes_file)  

    # Run DeepSpikeSort algorithm
    min_samples = int(sys.argv[2])
    max_samples = int(sys.argv[3])
    num_units = int(sys.argv[4])   
    noise_samples = int(sys.argv[5]) 
    
    selected_units = dataset.select_units(spikes, min_samples, max_samples, num_units=num_units, noise=True)
    spikes_dataset = dataset.SupervisedDataset(spikes_folder, selected_units, noise_samples=noise_samples)
    labels_map = spikes_dataset.labels_map
    
    # Calculate split sizes
    dataset_size = len(spikes_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(spikes_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64
    )  
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64
    )  
          
    features_flattened = 35328
    classifier_model = DeepCluster(num_units+1, features_flattened)     
   
    device=torch.device("cuda:0")
    num_devices = int(sys.argv[6])
    device_ids=list(range(num_devices))
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(classifier_model.parameters(), lr=0.0001)
        
    train_model = TrainModel(
        device, device_ids, 
        loss_fn, optimizer, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader
    )
    
    num_epochs = int(sys.argv[7])
    session_id = int(sys.argv[8])
    session_name = f'SUP_{session_id:03}'
    
    print("Running Classifier...")
    train_model.train_validate(classifier_model, num_epochs, labels_map, session_name, results_folder)
    
    del classifier_model
    torch.cuda.empty_cache()
