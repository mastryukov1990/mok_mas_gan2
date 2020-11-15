from torch import nn
import torch
from . import Conv2dBlock
from . import Dense2DBlock


class Discriminator(nn.Module):
    def __init__(self, params):
        super( Discriminator, self).__init__()
        self.global_branch = nn.Sequential(*[Conv2dBlock(pars) for pars in params["global_branch"]])
        self.dense_global_branch = nn.Sequential(*[Dense2DBlock(params["dense_global_branch"])  ])

        self.local_branch = nn.Sequential(*[Conv2dBlock(pars) for pars in params["local_branch"]])
        self.dense_local_branch = nn.Sequential(*[Dense2DBlock(params["dense_local_branch"])])

        self.neck = nn.Sequential(*[Dense2DBlock(params["neck"])])

    def forward(self, x, **kwargs):
        x1 = self.global_branch(x)

        x1 = self.dense_global_branch(x1.view(-1,2048))
        x2 = self.local_branch(x)

        x2 = self.dense_local_branch(x2.view(-1,2048))
        return self.neck(torch.cat([x1,x2],1))
