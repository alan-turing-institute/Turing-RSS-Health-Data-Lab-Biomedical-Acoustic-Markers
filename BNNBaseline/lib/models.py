import torch.nn as nn
import torch
from torchvision.models import resnet50
# from vggish.vggish import VGGish
import ResNetSource
# import ResNetSource
# from ResNetSource import resnet50dropout
import numpy as np
import torch.nn.functional as F # For dropout
import torch.nn.init as init

class ResnetDropoutFull(nn.Module):
    ''' PyTorch NN module for the full ResNet-50 Dropout network. The final linear layer is removed
        and replaced with the appropriate size linear layer, with output of a single unit to be 
        used with the BCELoss(). Note that some criterions may instead expect 2 units on the output'''
    def __init__(self, dropout=0.2):
#     def __init__(self):
        super(ResnetDropoutFull, self).__init__()
        self.resnet = ResNetSource.resnet50dropout(pretrained=True, dropout_p=0.2)
        self.dropout = dropout
        ##Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        
        self.fc1 = nn.Linear(2048, 1)
             
#         self.apply(_weights_init)
        
    def forward(self, x):      
        x = self.resnet(x).squeeze()
#         x = self.fc1(x)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        
        return x