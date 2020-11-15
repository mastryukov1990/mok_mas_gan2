from .Loss import Loss, MeanLoss
from torchvision.models import  vgg19
import torch
from torch import Tensor
from .utils import show_hiden, imshow, activ_map

class SGRL(Loss):
    def __init__(self, *args, **kwargs):
        super(SGRL, self).__init__()
        self.network = vgg19(True).features[:12]
        # print(self.network)

    def find_guid(self, inp, tar, *args, **kwargs):
        inp = inp.view(-1, 3, 128, 128)  # view to need size
        tar = tar.view(-1, 3, 128, 128)  # view to need size

        Mror = (inp - tar).mean(1).view(-1, 1, 128, 128)  # MSE - target shape [-1,1,H, W]
        Mror_min = torch.min(torch.min(Mror, 2)[0], 2)[0].view(-1, 1, 1, 1)  # min for evry image
        Mror_max = torch.max(torch.max(Mror, 2)[0], 2)[0].view(-1, 1, 1, 1)  # max for evry image

        return (Mror - Mror_min) / (Mror_max - Mror_min + 10 ** -8)  # centering and Max_Min_norm

    def find_active_maps(self, input, target, t_num, *args, **kwargs):
        return activ_map(self.network, input, target, t_num)

    def self_guid(self, maps, Mguid, *args, **kwargs):
        return maps * Mguid

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        maps = self.find_active_maps(input, target, 2)
        Mguid = self.find_guid(input, target)

        return torch.mean(torch.abs(self.self_guid(maps, Mguid)))