from .Loss import Loss, MeanLoss
from .SGRL import SGRL
from .AL import AdversialLoss
from .GL import GeometricLoss
from .FML import FML

class FinalLoss(Loss):
  def __init__(self):
    super(FinalLoss ,self).__init__()
    self.SGRL= SGRL()
    self.meanloss= MeanLoss()
    self.AL= AdversialLoss()
    self.GeometricLoss = GeometricLoss()
    self.FML = FML()

  def forward(self, x, y, labels, net):
    return self.SGRL(x, y) + self.meanloss(x, y)+self.AL(x, labels, net)  +self.GeometricLoss(x, y, )