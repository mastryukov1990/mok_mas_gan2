from .Loss import Loss
import torch
from torch import nn



class GAL(Loss):
    def __init__(self, type_a='dis'):
        super(GAL, self).__init__()
        self.a = 1 if type_a == 'dis' else -1

    def Dra(self, dis_X1, dis_X2, *args, **kwargs):
        # print(X1, X2)
        return nn.Sigmoid()(dis_X1 - torch.mean(dis_X2, 0))

    def forward(self, label, D_x, *args, **kwargs):
        # print(x[label==0].size())

        Drf = self.Dra(D_x[label == 1], D_x[label == 0]) * self.a
        Dfr = self.Dra(D_x[label == 0], D_x[label == 1]) * self.a
        # print(Drf.size())
        return -torch.mean(torch.log(1 - Dfr + 10 ** -10)) - torch.mean(torch.log(Drf + 10 ** -10))
