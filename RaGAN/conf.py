import torch
IMSIZE = 128
DEVICE = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
