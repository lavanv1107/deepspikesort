import os
import sys

import numpy as np

import torch
import torch.nn as nn

from pycave.bayes import GaussianMixture

import logging

import warnings
warnings.simplefilter("ignore")

import spikeinterface.full as si

import dataset
from model import DeepCluster
from deepspikesort import run_deepspikesort
import comparison
import util


if __name__ == "__main__":        
    recording_num = int(sys.argv[1])
    output_folder = f'output/{recording_num:03}/dss'
    results_folder = f'results/{recording_num:03}/dss'
    [os.makedirs(folder, exist_ok=True) for folder in [output_folder, results_folder]]
    
    # Load matched peaks object
    peaks_folder = f'peaks/{recording_num:03}'
    peaks_matched_file = os.path.join(peaks_folder, "peaks_matched.npy")
    peaks_matched = np.load(peaks_matched_file)  

    # Run DeepSpikeSort algorithm
    min_samples = int(sys.argv[2])
    max_samples = int(sys.argv[3])
    num_units = int(sys.argv[4])   
    
    selected_units = dataset.select_units(peaks_matched, min_samples, max_samples, num_units=num_units)
    peaks_dataset = dataset.UnsupervisedDataset(peaks_folder, selected_units)
        
    num_classes = int(sys.argv[5])    
    features_flattened = 35328
    deepcluster_model = DeepCluster(num_classes, features_flattened)     
    gmm = GaussianMixture(num_classes, covariance_type="full", init_strategy='kmeans', trainer_params=dict(gpus=[0])) 
    
    num_devices = int(sys.argv[6])
    deepcluster_kwargs = dict(
        dataset_folder=peaks_folder, 
        output_folder=output_folder, 
        results_folder=results_folder, 
        batch_size=64,
        sampling_frequency = 30000,
        device=torch.device("cuda:0"), 
        device_ids=list(range(num_devices)), 
        loss_fn=nn.CrossEntropyLoss(), 
        optimizer=torch.optim.Adam(deepcluster_model.parameters(), lr=0.0001),
        verbose_count=25
    )
    
    num_epochs = int(sys.argv[7])
    print("Running DeepSpikeSort...")
    dss_labels, dss_times = run_deepspikesort(
        peaks_dataset, 
        num_units, num_classes,
        deepcluster_model, gmm,
        num_epochs, deepcluster_kwargs
    )
    
    del deepcluster_model
    del gmm    
    torch.cuda.empty_cache()
    
    print('\n')
    
    # Create custom NumpySorting object from DeepSpikeSort output
    sorting_dss = comparison.create_numpy_sorting(dss_times, dss_labels, 30000)
    
    # Create custom NumpySorting object from matched peaks object
    mask_selected = np.isin(peaks_matched['unit_index'], [int(unit) for unit in selected_units]) # Create a boolean mask   
    peaks_selected = peaks_matched[mask_selected] # Filter the array
    peak_times = peaks_selected['sample_index']
    peak_units = peaks_selected['unit_index']
    sorting_peaks = comparison.create_numpy_sorting(peak_times, peak_units, 30000)
    
    # Log the comparison       
    print('Running SI comparison...')
    comparison.log_si_comparison(
        sorting_dss, sorting_peaks,
        'DeepSpikeSort', 'Peaks',
        num_units, peak_units, dss_labels,
        results_folder
    )
