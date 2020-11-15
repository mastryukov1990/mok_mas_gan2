import torch
import torch.nn as nn
from .Conv2dBlock import Conv2dBlock, NormActBlock
import itertools


class DMFB(nn.Module):
    def __init__(self, params: dict):
        super(DMFB, self).__init__()
        self.first_conv = Conv2dBlock(params['first_conv'])
        self.parallel_list = nn.ModuleList(
            [Conv2dBlock(pars) for pars in params['parallel']]
        )
        self.concat_list = nn.ModuleList(
            [Conv2dBlock(pars) for pars in params['concat']]
        )
        self.cat_norm_act = NormActBlock(params['cat_norm_act'])
        self.last_conv = Conv2dBlock(params['last_conv'])

    def __call__(self, x, **kwargs):
        return self.forward(x)

    def forward(self, x, **kwargs):
        y = self.first_conv(x)
        temp_output = [module(y) for module in self.parallel_list]
        for p, q, layer in zip(temp_output[:-1], temp_output[1:], self.concat_list):
            q = layer(p + q)
        combined = torch.cat(temp_output, dim=1)  # channels concat
        return x + self.last_conv(self.cat_norm_act(combined))

