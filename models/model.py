import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSpikeSort(nn.Module):
    def __init__(self, num_classes):
        super(DeepSpikeSort, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(9, 3, 2))
        self.bn1 = nn.BatchNorm3d(num_features=32) 
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(num_features=64) 
        self.drop2 = nn.Dropout2d()
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(35328, 5000)
        self.fc2 = nn.Linear(5000, num_classes) 
        
        # Initialize weights
        self._initialize_weights()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                

    def forward(self, x, feature_extraction=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(torch.squeeze(x, 4), 2)
        
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = self.drop2(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        
        if feature_extraction:
            return x  
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x