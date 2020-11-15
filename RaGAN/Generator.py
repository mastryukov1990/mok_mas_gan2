import torch.nn as nn
from .Conv2dBlock import Conv2dBlock
from .DMFB import DMFB


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(*[Conv2dBlock(pars) for pars in params["encoder"]])
        self.blocks = nn.Sequential(*[DMFB(pars) for pars in params["blocks"]])
        self.decoder = nn.Sequential(*[Conv2dBlock(pars) for pars in params["decoder"]])

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        x = self.blocks(x)
        return self.decoder(x)
