import torch
import torch.nn as nn
from layers.conv_layer import ConvLayer
from layers.fire_module import FireModule
from layers.pool_layers.maxpool_layer import MaxPoolLayer
from layers.pool_layers.avgpool_layer import AvgPoolLayer
from layers.flatten_layer import FlattenLayer

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvLayer(3, 96, kernel_size=7, stride=2, padding=3),  # conv1
            MaxPoolLayer(kernel_size=3, stride=2),                  # maxpool1
            
            FireModule(96, 16, 64, 64),                             # fire2
            FireModule(128, 16, 64, 64),                            # fire3
            FireModule(128, 32, 128, 128),                          # fire4
            MaxPoolLayer(kernel_size=3, stride=2),                  # maxpool4
            
            FireModule(256, 32, 128, 128),                          # fire5
            FireModule(256, 48, 192, 192),                          # fire6
            FireModule(384, 48, 192, 192),                          # fire7
            FireModule(384, 64, 256, 256),                          # fire8
            MaxPoolLayer(kernel_size=3, stride=2),                  # maxpool8
            
            FireModule(512, 64, 256, 256)                           # fire9
        )
        
        self.classifier = nn.Sequential(
            ConvLayer(512, num_classes, kernel_size=1),             # conv10
            AvgPoolLayer(kernel_size=13),                           # avgpool10
            FlattenLayer()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
