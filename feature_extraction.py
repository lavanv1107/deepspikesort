import os

import time

import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from util import AverageMeter 


def extract_features(model, dataloader, feature_dim, epoch, device, device_ids=None, verbose_count=10):    
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
        
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    
    num_samples = len(dataloader.dataset)
    
    features = torch.zeros(num_samples, feature_dim).to(device)
    
    total_batches = len(dataloader)
    verbose_interval = max(1, total_batches // verbose_count)
    
    end = time.time()
    start_idx = 0
    with torch.no_grad():
        for batch, inputs in enumerate(dataloader):
            data_time_meter.update(time.time() - end)
            inputs_dict = {'x': inputs, 'feature_extraction': True}
            outputs = model(inputs_dict)
            
            end_idx = start_idx + outputs.shape[0]
            features[start_idx:end_idx] = outputs
            start_idx = end_idx
            
            batch_time_meter.update(time.time() - end)
            end = time.time()
            
            if batch % verbose_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'
                      .format(epoch, batch, len(dataloader), batch_time=batch_time_meter,
                              data_time=data_time_meter))
            
    # Move final features tensor to CPU and convert to numpy
    features = features.cpu().numpy()
    
    return features


def preprocess_features(features, n_components):
    # Perform PCA dimensionality reduction
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(features)

    # Whitening
    scaler = StandardScaler()
    embeddings_whitened = scaler.fit_transform(embeddings_pca)

    # l2-Normalization
    embeddings_normalized = normalize(embeddings_whitened, norm='l2', axis=1)
    
    return embeddings_normalized


