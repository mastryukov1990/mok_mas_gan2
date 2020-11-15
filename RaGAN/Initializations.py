import numpy as np
import torch.nn as nn
import torch.tensor


def heInit(module: nn.Module, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if type(module) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)

