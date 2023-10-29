import os

import torch
import torch.nn as nn

from tqdm.notebook import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from collections import Counter


def extract_features(dataloader, feature_dim, model, device, device_ids=None):
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    
    num_samples = len(dataloader.dataset)
    
    features = torch.zeros(num_samples, feature_dim).to(device)
    folders = []
    
    start_idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Extracting Features", dynamic_ncols=True)):
            inputs_dict = {'x': inputs, 'feature_extraction': True}
            outputs = model(inputs_dict)
            
            end_idx = start_idx + outputs.shape[0]
            features[start_idx:end_idx] = outputs
            start_idx = end_idx
            
            dataset = dataloader.dataset.dataset
            
            # Get folder names using the get_image_folder method
            batch_folders = [
                dataset.get_label(idx)
                for idx in range(batch_idx * dataloader.batch_size, (batch_idx + 1) * dataloader.batch_size)
                if idx < len(dataloader.dataset)
            ]
            
            folders.extend(batch_folders)
            
    # Move final features tensor to CPU and convert to numpy
    features = features.cpu().numpy()
    
    return features, folders


def preprocess_features(features, n_components=100):
    # Perform PCA dimensionality reduction
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(features)

    # Whitening
    scaler = StandardScaler()
    embeddings_whitened = scaler.fit_transform(embeddings_pca)

    # l2-Normalization
    embeddings_normalized = normalize(embeddings_whitened, norm='l2', axis=1)
    
    return embeddings_normalized


