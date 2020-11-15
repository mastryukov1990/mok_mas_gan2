import torch.nn as nn
from typing import Tuple


class NormActBlock(nn.Module):
    def __init__(self, block_params):
        super(NormActBlock, self).__init__()
        self.block = nn.Sequential()
        if 'norm' in block_params:
            self.__add_normalization(block_params['norm'])
        if 'activation' in block_params:
            self.__add_activation(block_params['activation'])

    def forward(self, x, **kwargs):
        return self.block(x)

    def __add_normalization(self, norm_params: dict, **kwargs) -> None:
        self.block.add_module("normalization",
                              eval("nn." + norm_params["type"] +
                                   "(**norm_params[\"params\"])"))

    def __add_activation(self, act_params: dict, **kwargs) -> None:
        self.block.add_module("activation",
                              eval("nn." + act_params["type"] +
                                   "(**act_params[\"params\"] "
                                   "if 'params' in act_params else {})"))


class Conv2dBlock(nn.Module):
    def __init__(self, block_params):
        super(Conv2dBlock, self).__init__()
        transpose = block_params['transpose'] \
            if 'transpose' in block_params else False

        conv = nn.Conv2d(**block_params['conv']) \
            if not transpose else nn.ConvTranspose2d(**block_params['conv'])

        auto_pad = 'padding' not in block_params

        if auto_pad:
            conv.padding = Conv2dBlock.auto_padding(kernel_size=conv.kernel_size,
                                                    dilation_rate=conv.dilation)

        self.block = nn.Sequential(conv)
        if 'norm_act' in block_params:
            self.block.add_module("norm_act", NormActBlock(block_params["norm_act"]))

    def __call__(self, x, **kwargs):
        return self.forward(x)

    @staticmethod
    def auto_padding(kernel_size, dilation_rate):
        return ((kernel_size[0] - 1) // 2 * dilation_rate[0],
                (kernel_size[1] - 1) // 2 * dilation_rate[1])

    def forward(self, x, **kwargs):
        return self.block(x)


