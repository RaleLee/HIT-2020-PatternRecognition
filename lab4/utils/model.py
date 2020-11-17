import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelManager(nn.Module):

    def __init__(self, args, num_label):
        super(ModelManager, self).__init__()
        self.__args = args
        self.__num_label = num_label

    def forward(self, vector):
        pass
