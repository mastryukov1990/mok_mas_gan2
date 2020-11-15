from torch import nn
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