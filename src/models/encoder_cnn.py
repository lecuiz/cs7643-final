import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import ast

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

class EncoderCNN(nn.Module):
    """
    The Visual Backbone (ResNet-101)
    """
    def __init__(self, embed_dim):
        super(EncoderCNN, self).__init__()
        # Load pretrained ResNet-101
        try:
            from torchvision.models import ResNet101_Weights
            resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        except (ImportError, AttributeError):
            resnet = models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)
        features = self.embed(features)
        return features
