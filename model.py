import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepCluster(nn.Module):
    def __init__(self, num_classes, features_flattened):
        super(DeepCluster, self).__init__()
        # Extractor layers
        self.conv_layer_1 = nn.Conv3d(1, 32, kernel_size=(9, 3, 2)) 
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4) 
        self.conv_layer_2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        
        # Classifier layers
        self.fully_connected_layer_1 = nn.Linear(features_flattened, 500) # figure out a suitable number of features
        self.fully_connected_layer_2 = nn.Linear(500, num_classes)
        
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

    def forward(self, inputs_dict):
        x = inputs_dict['x']
        feature_extraction = inputs_dict.get('feature_extraction', False)
        
        # Extractor 
        x = F.relu(F.max_pool2d(torch.squeeze(self.conv_layer_1(x), 4), 2))
        x = F.relu(F.max_pool2d(self.conv_layer_2_drop(self.conv_layer_2(x)), 2))
        
        x = self.flatten(x)
        
        x = F.relu(self.fully_connected_layer_1(x))
        
        if feature_extraction:
            return x  # Returning here before dropout and the second fully connected layer
        
        # Classifier
        x = F.dropout(x, training=self.training)
        x = self.fully_connected_layer_2(x)
        
        return x


class Extractor(nn.Module):
    def __init__(self, features_flattened):
        super().__init__()
        self.conv_layer_1 = nn.Conv3d(1, 32, kernel_size=(9, 3, 2)) 
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4) 
        
        self.conv_layer_2_drop = nn.Dropout2d()
        
        self.flatten = nn.Flatten()
        self.fully_connected_layer_1 = nn.Linear(features_flattened, 500)
        
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(F.max_pool2d(torch.squeeze(self.conv_layer_1(x), 4), 2))
        x = F.relu(F.max_pool2d(self.conv_layer_2_drop(self.conv_layer_2(x)), 2))
        
        x = self.flatten(x)
        x = F.relu(self.fully_connected_layer_1(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class Classifier(nn.Module):
    def __init__(self, num_classes, features_flattened):
        super().__init__()
        self.conv_layer_1 = nn.Conv3d(1, 32, kernel_size=(9, 3, 2)) 
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4) 
        
        self.conv_layer_2_drop = nn.Dropout2d()
        
        self.flatten = nn.Flatten()
        self.fully_connected_layer_1 = nn.Linear(features_flattened, 500)
        self.fully_connected_layer_2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(torch.squeeze(self.conv_layer_1(x), 4), 2))
        x = F.relu(F.max_pool2d(self.conv_layer_2_drop(self.conv_layer_2(x)), 2))
        
        x = self.flatten(x)
        x = F.relu(self.fully_connected_layer_1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fully_connected_layer_2(x))
        return x