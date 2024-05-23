import argparse
import os
import sys
import warnings

import numpy as np

import torch
import torch.nn as nn

from accelerate import Accelerator

# from pycave.bayes import GaussianMixture
# from cuml.cluster import HDBSCAN

from .deepspikesort import DeepSpikeSortPipeline

sys.path.append("..")
from load_dataset import select_units, TraceDataset
from model import DeepSpikeSort

warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Process training session parameters.')

    # Retrieve parameters from command line arguments
    parser.add_argument('recording_id', type=str, help='ID of the recording')
    parser.add_argument('min_samples', type=int, help='Minimum number of samples per unit')
    parser.add_argument('max_samples', type=determine_count, help='Maximum number of samples per unit')
    parser.add_argument('num_units', type=determine_count, help='Number of units')
    parser.add_argument('num_samples', type=determine_count, help='Number of samples per unit')
    parser.add_argument('noise_samples', type=determine_count, help='Number of noise samples')
    parser.add_argument('session_name', type=str, help='Name of the training session')
    parser.add_argument('session_number', type=int, help='Number of the training session')

    return parser.parse_args()
    
    
def main(args):
    # Load peaks data from the file
    peaks_folder = f'data/{args.recording_id}/peaks'
    peaks_file = os.path.join(peaks_folder, "peaks_matched.npy")
    peaks = np.load(peaks_file)  

    # Select units based on the parameters
    # selected_units = select_units(peaks, min_samples=args.min_samples, max_samples=args.max_samples, num_units=args.num_units)
    selected_units = [150, 310, 245] # selected units for testing
    
    # Create an unsupervised trace dataset from selected units
    peaks_dataset = TraceDataset(peaks_folder, 'unsupervised', selected_units, num_samples=args.num_samples, noise_samples=args.noise_samples)
    
    # Define the CNN model
    cnn = DeepSpikeSort(num_units)   
    
    # Define the clustering model
    # gmm = GaussianMixture(num_units, covariance_type="full", init_strategy='kmeans', trainer_params=dict(gpus=[0]))   
    # hdb = HDBSCAN(min_cluster_size=30)
    
    # Specify loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()    
    optimizer = torch.optim.Adam(dss_model.parameters(), lr=0.0001)    
    
    # Initialize the Hugging Face Accelerator
    accelerator = Accelerator()

    # Create a directory for output 
    output_folder = f'phase_2/output/{args.recording_id}'
    os.makedirs(output_folder, exist_ok=True)
    
    # Run DeepSpikeSort
    session_id = f'{args.session_name.upper()}_{args.session_number:03}' # Set an ID for the training session
    dss = DeepSpikeSortPipeline(peaks_folder, peaks_dataset, cnn, loss_fn, optimizer, accelerator, cluster_model=None, output_folder, session_id)
    
    if accelerator.is_main_process:
        print("Running DeepSpikeSort...")
        
    dss.run_deepspikesort(100)
                        
    # Free up memory by deleting the model and clearing CUDA cache
    del deepcluster_model
    torch.cuda.empty_cache()

    
def determine_count(value):
    if value == 'max' or value=='all':
        return value
    else:
        return int(value)

    
if __name__ == "__main__":        
    args = parse_args()
    main(args)
