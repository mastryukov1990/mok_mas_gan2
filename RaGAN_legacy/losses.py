import torch
from torchvision.models import vgg19
from utils import show_hidden, imshow, activ_map
from conf import IMSIZE, DEVICE


from typing import List, Dict, Set, Tuple, Any, Optional


class Loss:
    def __init__(self, ):
        self.history = []
        self.last_meaning = None

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass


class MeanLoss(Loss):
    def __init__(self, ):
        super(MeanLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(input - target).mean()
