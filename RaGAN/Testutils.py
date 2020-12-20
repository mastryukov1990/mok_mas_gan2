from PIL import Image
from torchvision import transforms
from .conf import IMSIZE, DEVICE
import torch

img = Image.open('256_256.jpg')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IMSIZE, IMSIZE)),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
img = transform(img)
wimg = torch.stack([img.view(3, IMSIZE, IMSIZE), img.view(3, IMSIZE, IMSIZE)]).to(DEVICE)  # example pic
