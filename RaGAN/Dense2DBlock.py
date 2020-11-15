from torch import nn
from .NormActBlock import NormActBlock

class Dense2DBlock(nn.Module):
  def __init__(self, block_params):
    super(Dense2DBlock, self).__init__()
    dense = nn.Linear( **block_params['linear'] )
    self.block = nn.Sequential(dense)
    if 'norm_act' in block_params:
      self.block.add_module('act', NormActBlock(block_params["norm_act"]))

  def forward(self, x):
    return self.block(x)


