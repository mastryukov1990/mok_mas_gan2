from torch import nn
import torch
from .Conv2dBlock import Conv2dBlock
from .Dense2DBlock import Dense2DBlock


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.global_branch = nn.Sequential(*[Conv2dBlock(pars) for pars in params["global_branch"]])
        self.dense_global_branch = nn.Sequential(*[Dense2DBlock(params["dense_global_branch"])])

        self.local_branch = nn.Sequential(*[Conv2dBlock(pars) for pars in params["local_branch"]])
        self.dense_local_branch = nn.Sequential(*[Dense2DBlock(params["dense_local_branch"])])

        self.neck = nn.Sequential(*[Dense2DBlock(params["neck"])])
        self.last_act = nn.Sigmoid()

    def forward(self, gt, masked, sig=True, **kwargs):

        x1 = self.global_branch(gt)
        x1 = self.dense_global_branch(x1.view(x1.shape[0], -1))

        x2 = self.local_branch(masked)
        x2 = self.dense_local_branch(x2.view(x2.shape[0], -1))

        y = self.neck(torch.cat([x1, x2], 1))
        if sig:
            y = self.last_act(y)
        return y
