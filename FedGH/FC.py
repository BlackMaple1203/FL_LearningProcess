from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np
import math
import torch

class FC(nn.Module):
    def __init__(self, in_dim = 500, out_dim = 10):
        super(FC, self).__init__()

        self.fc3 = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        o = self.fc3(x)
        return o