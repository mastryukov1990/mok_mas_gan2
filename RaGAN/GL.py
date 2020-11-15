from .Loss import Loss
from .utils import give_net
import numpy as np
from torchvision.models import vgg19
import torch

class GeometricLoss(Loss):
    def __init__(self, *args, **kwargs):
        super(GeometricLoss, self).__init__()
        self.net = give_net(vgg19(True).features[:12], 5)

    def forward(self, output, target, *args, **kwargs):
        output = self.net(output)
        target = self.net(target)

        vectorH = torch.Tensor([i for i in np.arange(output.size()[-1])])
        vectorW = torch.Tensor([i for i in np.arange(output.size()[-2])])

        d_o = output @ vectorH / output.sum()
        w_o = vectorW @ output / output.sum()
        d_t = target @ vectorH / target.sum()
        w_t = vectorW @ target / target.sum()

        return ((d_t - d_o) ** 2 + (w_t - w_o) ** 2).sum()