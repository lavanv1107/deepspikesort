import argparse
import datetime
import gc
import os
import sys
import warnings

import numpy as np

import torch
import torch.nn as nn

from accelerate import Accelerator

from deepspikesort import DeepSpikeSortPipeline

sys.path.append("..")
import load_dataset
from models import model

warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Process training session parameters.")

    # Retrieve parameters from command line arguments
    parser.add_argument("recording_id", type=str, help="ID of the recording")
    parser.add_argument("min_samples", type=int, help="Minimum number of samples per unit")
    parser.add_argument("max_samples", type=determine_count, help="Maximum number of samples per unit")
    parser.add_argument("num_units", type=determine_count, help="Number of units")
    parser.add_argument("seed", type=int, help="Seed for random selection of units")
    parser.add_argument("num_samples", type=determine_count, help="Number of samples per unit")
    parser.add_argument("noise_samples", type=determine_count, help="Number of noise samples")
    parser.add_argument("method", type=str, help="Method to handle coincedent spikes")    
    parser.add_argument("trial_name", type=str, help="Name of the training trial")
    parser.add_argument("trial_number", type=int, help="Number of the training trial")

    return parser.parse_args()
    
    
def main(args):    
    # Initialize the Hugging Face Accelerator
    accelerator = Accelerator()
    
    # Load peaks data from the file
    peaks_folder = f"../data/{args.recording_id}/peaks"
    peaks_file = os.path.join(peaks_folder, "peaks_matched.npy")
    peaks = np.load(peaks_file)          
    
    channels_file = f"../data/{args.recording_id}/channel_locations.npy"
    channels = np.load(channels_file)      
    
    if accelerator.is_main_process:
        print("Preparing dataset...")

    # Select units based on the parameters
    units_selected = load_dataset.select_units(peaks, num_units=args.num_units, min_samples=args.min_samples, max_samples=args.max_samples, seed=args.seed)
    
    # Create an unsupervised trace dataset from selected units
    peaks_dataset = load_dataset.TraceDataset(
        peaks_folder, 'eval',
        units_selected, num_samples=args.num_samples, noise_samples=args.noise_samples,
        channels=channels, method=args.method
    )
    
    num_units = args.num_units
    
    if args.noise_samples != 0:
        num_units = args.num_units + 1
        units_selected = np.append(units_selected, -1)
    
    batch_size = 32
    
    if accelerator.is_main_process:
        print("Building pipeline...")
    
    # Define the CNN model
    cnn = model.DeepSpikeSort(num_units)  
    
    # Specify loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()    
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)    
    
    # Create a directory for output 
    output_folder = f"phase_2/output/{args.recording_id}"
    os.makedirs(output_folder, exist_ok=True)
    
    session_date = datetime.datetime.now().strftime("%Y-%m-%d")
    session_folder = os.path.join(output_folder, session_date)
    os.makedirs(session_folder, exist_ok=True)
    
    trial_id = f"{args.trial_name.upper()}_{args.trial_number:03}" 
    trial_folder = os.path.join(session_folder, trial_id)
    os.makedirs(trial_folder, exist_ok=True)
    
    trial_info_file = os.path.join(trial_folder, f"{trial_id}_info.txt")
    with open(trial_info_file, "w") as file:
        file.write("Minimum samples per unit: {0}\n"
                   "Maximum samples per unit: {1}\n"
                   "Number of units: {2}\n"
                   "Random seed: {3}\n"
                   "Number of samples to use per unit: {4}\n"
                   "Number of noise samples to add: {5}\n"
                   "Method used: {6}\n" 
                   "Selected units: \n{7}"
                   .format(args.min_samples, 
                           args.max_samples, 
                           num_units, 
                           args.seed,
                           args.num_samples,
                           args.noise_samples,
                           args.method,
                           units_selected))
        
    np.save(os.path.join(trial_folder, f"{trial_id}_units_selected.npy"), units_selected)
    
    # Run DeepSpikeSort
    dss = DeepSpikeSortPipeline(
        peaks_dataset, batch_size,
        cnn, loss_fn, optimizer, accelerator, num_units, 
        trial_folder, trial_id
    )
        
    if accelerator.is_main_process:
        print("Running DeepSpikeSort...")
        
    num_epochs = 100
    dss.run_deepspikesort(num_epochs)
                        
    # Free up memory 
    del cnn
    torch.cuda.empty_cache()
    
    # Run garbage collector
    gc.collect()

    
def determine_count(value):
    if value == "max" or value=="all":
        return value
    else:
        return int(value)

    
if __name__ == "__main__":        
    args = parse_args()
    main(args)