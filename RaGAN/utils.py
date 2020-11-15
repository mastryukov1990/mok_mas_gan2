from .conf import IMSIZE
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


def activ_map(net, img, tar, t_num):
    img = img.view(-1, 3, IMSIZE, IMSIZE)  # view to need size
    tar = tar.view(-1, 3, IMSIZE, IMSIZE)  # view to need size
    num = 0  # num of ReLU  now
    tens = []
    for layer in net:
        img = layer(img)  # process imgs
        tar = layer(tar)
        if isinstance(layer, nn.ReLU):
            num += 1
            tens.append(torch.sum((img - tar), 1).view(-1, 1, 128, 128) / img.size()[1] ** 2)
            if num == t_num:
                return torch.stack(tens)


def show_hiden(net, img):
    images = {}
    num = 0
    img = img.view(-1, 3, IMSIZE, IMSIZE)

    # print(net, img.size())
    for layer in net:
        img = layer(img)
        if isinstance(layer, nn.ReLU):
            print(layer)
            num += 1
            arr = np.array(img.detach().mean(1))
            arr_ = np.squeeze(arr)
            images[num] = arr_
    return images


def give_net(net, t_num, *args, **kwargs):
    num = 0
    num_layers = 0
    new_net = []
    mod = nn.Sequential()

    for layer in net:
        num_layers += 1
        layer
        if isinstance(layer, nn.ReLU):
            num += 1
            if num == t_num:
                return net[:num_layers]


def give_blocks(net, num_act):
    modules = nn.ModuleList()
    module = nn.Sequential()
    for i, layer in enumerate(net):
        module.add_module('{i}_{name}'.format(i=i, name=layer.__class__.__name__), layer)
        if isinstance(layer, nn.ReLU):
            modules.append(module)
            module = nn.Sequential()
            num_act -= 1
        if num_act == 0:
            return modules


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()
