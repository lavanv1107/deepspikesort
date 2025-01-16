# Deep Spike Sorting Pipeline

## Spike Sorting
Due to recent advances in electrode technology, neuroscientists can now record neuronal activity from hundreds of neurons simultaneously over minutes to hours. To make sense of the raw electrical signals, scientists must process the data to remove noise, detect spike events, and assign those spike events to different neurons based on the spike waveform shape and other properties. This process is known as “spike sorting”. 

This is a historically challenging problem; however, the latest methods in deep learning on audio and other time series data show promise in denoising and identifying repeated events.

## DeepCluster
DeepCluster is an unsupervised learning algorithm developed by Facebook AI researchers for clustering and representation learning. The algorithm alternates between clustering a dataset and fine-tuning a neural network using the obtained cluster assignments as pseudo-labels. The process consists of the following steps:
- Perform a forward pass on the unlabeled dataset using the current neural network to obtain feature embeddings.
- Cluster these feature embeddings into groups.
- Use the cluster assignments as pseudo-labels and update the neural network weights through backpropagation.

## Objective
The objective of this project is to develop and evaluate components of an automated, end-to-end, machine-learning-based spike sorting pipeline following the DeepCluster framework with the potential to evolve into a valuable resource for the broader neuroscience community.


## Create the dataset

Run the notebook `create_dataset/create_ibl_dataset1.ipynb` to create the dataset. This will create a folder 
called `data/sub-CSHL049` with the dataset.


## Phase 1: Classification with CNN

Run classification on spikes using `run_classify.py` from the main folder (or `submit_run_classify.sh` with SLURM).

You will need to specify 8 arguments:
- [1] The ID of the recording
- [2] The minimum number of samples per unit
- [3] The maximum number of samples per unit ('max' for the max number of samples per unit)
- [4] The number of units to be classified
- [5] The number of samples to be used per unit ('all' to use all samples)
- [6] The number of noise samples to include (0 for none, 'all' for all)
- [7] The name to set the session ID
- [8] The number to set the session ID

Example: `python -m phase_1.run_classify sub-CSHL049 1000 5000 3 all 0 sup 0`

The example command will run:
- using recording sub-CSHL049
- on 3 units
- with 1000-5000 samples per unit
- using all samples per unit
- including 0 noise samples
- named session SUP_000

The script will also save the classification results to the results folder:
- Accuracy and Loss plot
- Accuracy and Loss log
- Confusion matrix 

## Phase 2:

Install Pytorch following their instructions [here](https://pytorch.org/get-started/locally/).

Install accelerate: `pip install accelerate`


## TODO:
- Separate preprocessing of Phase 2 dataset (currently in `phase_2/dss_phase_2.ipynb`) into its own script
- Resolve issues with distributed training (look into `deepspikesort.py`)
- Resolve memory errors when loading data into GPU for training (look into `deepspikesort.py`)
