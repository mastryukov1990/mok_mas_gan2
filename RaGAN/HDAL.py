from torch.nn import ReLU
from torch import mean as mn
from .Loss import Loss


class HGAL(Loss):
    def __init__(self):
        super(HGAL, self).__init__()

    def forward(self, y_pred_real, y_pred_fake):
        return mn(ReLU()(1.0 - (y_pred_real - mn(y_pred_fake)))+ReLU()(1.0 + (y_pred_fake - mn(y_pred_real))))/2
