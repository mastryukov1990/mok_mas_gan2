from .Loss import Loss
import torch
from torch import nn


class AdversialLoss(Loss):
    def __init__(self, ):
        super(AdversialLoss, self).__init__()

    def Dra(self, dis_X1, dis_X2, *args, **kwargs):
        # print(X1, X2)
        return nn.Sigmoid()(dis_X1 - torch.mean(dis_X2, 0))

    def forward(self, label, D_x, *args, **kwargs):
        # print(x[label==0].size())
        Drf = self.Dra(D_x[label == 1], D_x[label == 0])
        Dfr = self.Dra(D_x[label == 0], D_x[label == 1])
        return -torch.mean(torch.log(Dfr - 10 ** -10)) - torch.mean(torch.log(1 - Drf + 10 ** -10))
