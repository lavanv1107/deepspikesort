import argparse
import os
import sys
import warnings

import numpy as np

import torch
import torch.nn as nn

from accelerate import Accelerator

from .classify import ClassifyPipeline

sys.path.append("..")
from data import load_dataset
from models import model

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
    # Load spikes data from file
    spikes_folder = f'data/{args.recording_id}/spikes'
    spikes_file = os.path.join(spikes_folder, "spikes.npy")
    spikes = np.load(spikes_file)  

    # Select units based on the parameters
    selected_units = load_dataset.select_units(spikes, min_samples=args.min_samples, max_samples=args.max_samples, num_units=args.num_units)
    
    # Create a supervised trace dataset from selected units
    spikes_dataset = load_dataset.TraceDataset(spikes_folder, 'supervised', selected_units, num_samples=args.num_samples, noise_samples=args.noise_samples)
    
     # Define the CNN model
    cnn = model.DeepSpikeSort(len(selected_units))
    
    # Specify loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()    
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)    
    
    # Initialize the Hugging Face Accelerator
    accelerator = Accelerator()

    # Create a directory for output 
    output_folder = f'phase_1/output/{args.recording_id}'
    os.makedirs(output_folder, exist_ok=True)
    
    # Run classification
    session_id = f'{args.session_name.upper()}_{args.session_number:03}' # Set an ID for the training session
    classify_pipeline = ClassifyPipeline(spikes_dataset, cnn, loss_fn, optimizer, accelerator, output_folder, session_id)
    
    if accelerator.is_main_process:
        print("Running Classification...")
        
    classify_pipeline.train_validate(100)

    # Free up memory by deleting the model and clearing CUDA cache
    del cnn
    torch.cuda.empty_cache()
    

def determine_count(value):
    if value == 'max' or value=='all':
        return value
    else:
        return int(value)
    
    
if __name__ == "__main__":        
    args = parse_args()
    main(args)
