from .Loss import Loss
from .utils import give_net, give_blocks
import numpy as np
from torchvision.models import vgg19
import torch
from torch import nn


class AdversialLoss(Loss):
    def __init__(self, ):
        super(AdversialLoss, self).__init__()

    def Dra(self, X1, X2, dis, *args, **kwargs):
        X1 = 0 if len(X1) == 0 else dis(X1)
        X2 = 0 if len(X2) == 0 else torch.mean(dis(X2), 0)
        return nn.Sigmoid()(X1 - X2)

    def forward(self, x, label, dis, *args, **kwargs):
        # print(x[label==0].size())
        Drf = self.Dra(x[label == 1], x[label == 0], dis)
        Dfr = self.Dra(x[label == 0], x[label == 1], dis)
        return -torch.mean(torch.log(Dfr - 10 ** -10)) - torch.mean(torch.log(1 - Drf + 10 ** -10))
