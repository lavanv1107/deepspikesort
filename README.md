# Deep Spike Sorting Pipeline
Due to recent advances in electrode technology, neuroscientists can now record neuronal activity from hundreds of neurons simultaneously over minutes to hours. To make sense of the raw electrical signals, scientists must process the data to remove noise, detect spike events, and assign those spike events to different neurons based on the spike waveform shape and other properties. This process is known as “spike sorting”. 

This is a historically challenging problem; however, the latest methods in deep learning on audio and other time series data show promise in denoising and identifying repeated events.

The objective of this project is to develop and evaluate components of an automated, end-to-end, machine-learning-based spike sorting pipeline with the potential to evolve into a valuable resource for the broader neuroscience community.

We utilized the following core libraries:
- SpikeInterface for for streamlined extracellular data file management and data processing 
- PyTorch for machine learning tasks
