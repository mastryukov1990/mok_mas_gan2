from .Loss import Loss
from .utils import give_net, give_blocks
import numpy as np
from torchvision.models import vgg19
import torch
from .conf import DEVICE


class FML(Loss):
    def __init__(self, *args, **kwargs):
        super(FML, self, ).__init__()

    @staticmethod
    def norm1(x, y):
        return torch.mean(abs(x - y))

    def forward(self, x, y, net, *args, **kwargs):
        error = torch.zeros([1]).to(DEVICE)
        for block in net:
            x, y = block(x), block(y)
            error += FML.norm1(x, y)
        return error
