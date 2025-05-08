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
    parser.add_argument("seed", type=int, help="")   
    parser.add_argument("method", type=str, help="Method to handle coincedent spikes")  
    parser.add_argument("trial_name", type=str, help="Name of the training trial")
    parser.add_argument("trial_number", type=int, help="Number of the training trial")

    return parser.parse_args()
    
    
def main(args):    
    # Initialize the Hugging Face Accelerator
    accelerator = Accelerator()
    
    peaks_folder = f"../data/{args.recording_id}/peaks"
    
    channel_locations_file = f"../data/{args.recording_id}/channel_locations.npy"
    channel_locations = np.load(channel_locations_file)      
    
    if accelerator.is_main_process:
        print("Preparing dataset...")

    # Create a trace dataset 
    peaks_dataset = load_dataset.TraceDataset(
        dataset_folder=peaks_folder, seed=args.seed, channel_locations=channel_locations, method=args.method
    )
    
    batch_size = 32
    
    if accelerator.is_main_process:
        print("Building pipeline...")
    
    # Define the CNN model
    num_units = 100
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
        file.write("Random seed: {0}\n"
                   "Method used: {1}\n" 
                   .format(args.seed,
                           args.method))
    
    # Run DeepSpikeSort
    dss = DeepSpikeSortPipeline(
        peaks_dataset, batch_size,
        cnn, loss_fn, optimizer, accelerator, num_units, 
        trial_folder, trial_id
    )
        
    if accelerator.is_main_process:
        print("Running DeepSpikeSort...")
        
    num_epochs = 25
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
