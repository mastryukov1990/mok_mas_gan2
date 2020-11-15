from torch import Tensor
import torch

class Loss:
    def __init__(self, ):
        self.history = []
        self.last_meaning = None

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        pass


class MeanLoss(Loss):
    def __init__(self, ):
        super(MeanLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.abs(input - target).mean()