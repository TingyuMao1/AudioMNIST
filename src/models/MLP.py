import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import lrelu


class MLP(nn.Module):

    def __init__(self, layers, dropout):
        """
        @param 
            layers: list of number of neurons in each layer, 
            both input and output layers are included
        """
        super(MLP, self).__init__()
        self.name = "MLP"
        self.dropout = dropout

        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.fc = [self.fc1, self.fc2, self.fc3]

        """
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
        """

    def forward(self, x):
        for i in range(len(self.fc)):
            x = lrelu(self.fc[i](x))
            x = F.dropout(x, training=self.training)
        x = F.log_softmax(x)
        return x

