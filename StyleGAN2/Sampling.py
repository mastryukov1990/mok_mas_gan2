import torch
import torch.nn as nn

<<<<<<< HEAD
from .op import upfirdn2d
=======
from .simple_op import upfirdn2d
>>>>>>> main


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    # transform to 2d
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    # normalize
    k /= k.sum()
    return k


class Upsampler(nn.Module):
    def __init__(self, kernel, scale=2):
        super(Upsampler, self).__init__()
        self.scale = scale
        kernel = make_kernel(kernel) * (scale ** 2)
        # we don't need kernel as a module parameter
        self.register_buffer("kernel", kernel)

        padding = kernel.shape[0] - scale
        padding0 = (padding + 1) // 2 + scale - 1
        padding1 = padding // 2
        self.pad = (padding0, padding1)

    def forward(self, input, **kwargs):
        return upfirdn2d(input, self.kernel, up=self.scale, down=1, pad=self.pad)


class Downsampler(nn.Module):
    def __init__(self, kernel, scale=2):
        super(Downsampler, self).__init__()

        self.scale = scale
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        padding = kernel.shape[0] - scale
        pad0 = (padding + 1) // 2
        pad1 = padding // 2
        self.pad = (pad0, pad1)

    def forward(self, input, **kwargs):
        return upfirdn2d(input, self.kernel, up=1, down=self.scale, pad=self.pad)




