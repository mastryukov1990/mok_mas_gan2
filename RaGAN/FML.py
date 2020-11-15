from .Loss import Loss
from .utils import give_net, give_blocks
import numpy as np
from torchvision.models import vgg19
import torch


class FML(Loss):
    def __init__(self, net, n, *args, **kwargs):
        super(FML, self, ).__init__()
        self.net = give_blocks(net, n)

    @staticmethod
    def norm1(x, y):
        return torch.mean(abs(x - y))

    def forward(self, x, y, *args, **kwargs):
        error = torch.zeros([1])
        for block in self.net:
            x, y = block(x), block(y)
            error += FML.norm1(x, y)
        return error
