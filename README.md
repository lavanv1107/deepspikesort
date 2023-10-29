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
