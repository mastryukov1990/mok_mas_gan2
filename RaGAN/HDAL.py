from torch.nn import ReLU, Tanh
from torch import mean as mn
from .Loss import Loss


class HDAL(Loss):
    def __init__(self):
        super(HDAL, self).__init__()

    def forward(self, y_pred_real, y_pred_fake):
        return mn(ReLU()(1.0 - Tanh()(y_pred_real - mn(y_pred_fake)))+ReLU()(1.0 + Tanh()(y_pred_fake - mn(y_pred_real))))/2
