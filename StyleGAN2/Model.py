import math
import random
import functools
import operator

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from .Sampling import make_kernel, Upsampler, Downsampler

kEpsilon = 1e-8


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x, **kwargs):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + kEpsilon)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_f=1):
        super(Blur, self).__init__()
        kernel = make_kernel(kernel)

        if upsample_f > 1:
            kernel = kernel * (upsample_f ** 2)

        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, x, **kwargs):
        return upfirdn2d(x, self.kernel, pad=self.pad)


class NormalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
                 ):
        super(NormalConv2d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.norm = 1 / math.sqrt(in_channels * (kernel_size ** 2))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x, **kwargs):
        return F.conv2d(
            x,
            self.weight * self.norm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1], self.weight.shape[0],},'
            f'{self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class NormalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bias_init=0, lr_mul=1.0, activation=None):
        super(NormalLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation
        self.scale = (1 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = lr_mul
        pass

    def forward(self, x, **kwargs):
        if self.activation == 'fused_lrelu' or self.activation:
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1], self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()

        self.negative_slope = negative_slope

    def forward(self, x, **kwargs):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=None
    ):
        super(ModulatedConv2d, self).__init__()

        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample
        self.eps = kEpsilon

        fan_in = in_channels * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = NormalLinear(style_dim, kernel_size, kernel_size)
        self.demodulate = demodulate

        assert not (upsample and downsample)
        if upsample:
            factor = 2
            pad = len(blur_kernel) - factor - kernel_size + 1
            pad0 = (pad + 1) // 2 + factor - 1
            pad1 = pad // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_f=factor)
        elif downsample:
            factor = 2
            pad = len(blur_kernel) - factor + kernel_size - 1
            pad0 = (pad + 1) // 2
            pad1 = pad // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_f=factor)

    def forward(self, x, style):
        batch, in_channels, in_h, in_w = x.shape
        style = self.modulation(style).view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum(2, 3, 4) + kEpsilon)
            weight = weight * d.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size)

        if self.upsample:
            x = x.view(1, batch * in_channels, in_h, in_w)
            weight = weight.view(batch, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, out_h, out_w = out.shape
            out = out.view(batch, self.out_channels, in_h, in_w)
            out = self.blur(out)
        elif self.downsample:
            x = self.blur(x)
            _, _, in_h, in_w = x.shape
            x = x.view(1, batch * in_channels, in_h, in_w)
            out = F.conv2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, out_h, out_w = out.shape
            out = out.view(batch, self.out_channels, out_h, out_w)
        else:
            x = x.view(1, batch * in_channels, in_h, in_w)
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            _, _, out_h, out_w = out.shape
            out = out.view(batch, self.out_channels, out_h, out_w)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size}, '
            f'upsampler={self.upsample}, downsampler={self.downsample})'
        )


class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch_size, _, in_h, in_w = image.shape
            noise = image.new_empty(batch_size, 1, in_h, in_w).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channels, size=4):
        super(ConstantInput, self).__init__()
        self.output = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.output.repeat(batch_size, 1, 1, 1)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super(ToRGB, self).__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels=3, kernel_size=1, style_dim=style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 3, 1))
        if upsample:
            self.upsampler = Upsampler(blur_kernel)

    def forward(self, x, style, skip=None):
        out = self.conv(input, style)
        out += self.bias

        if skip is not None:
            skip = self.upsampler(skip)
            out += skip

        return out


class StyledConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_dim,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True):
        super(StyledConv, self).__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate
        )

        self.noise = NoiseInjection()
        self.activation = FusedLeakyReLU(channel=out_channels)

    def forward(self, x, style, noise=None):
        out = self.conv(x, style=style)
        out = self.noise(out, noise=noise)
        out = self.activation(out)
        return out


class Generator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 style_dim=512,
                 mlp_no=8,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 fmap_base=16 << 10,  # 2 ** 14
                 fmap_decay=1.0,
                 fmap_min=1,
                 fmap_max=512,
                 start_resolution=4
                 ):
        resol_log = int(np.log2(resolution))
        start_res_log = int(np.log2(start_resolution))
        assert resolution == 2 ** resol_log and resolution >= 4
        assert start_resolution == 2 ** start_res_log
        super(Generator, self).__init__()

        num_layers = resol_log - int(np.log2(start_resolution))
        self.num_layers = num_layers * 2 + 1
        self.num_latent = self.num_layers + 1

        self.start_res_log = start_res_log
        self.start_resol = start_resolution
        self.resolution = resolution
        self.resol_log = resol_log

        self.style_dim = style_dim
        self.convs = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        mlp_layers = [PixelNorm()]
        for i in range(mlp_no):
            mlp_layers.append(
                NormalLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_relu'
                )
            )

        self.styler = nn.Sequential(*mlp_layers)
        self.channels = {l_number ** 2: self.__nf(l_number, fmap_base, fmap_decay, fmap_min, fmap_max) for l_number in
                         range(start_res_log, resol_log + 1)}

        self.input = ConstantInput(self.channels[start_resolution])
        self.conv_1 = StyledConv(
            self.channels[start_resolution],
            self.channels[start_resolution],
            kernel_size=3,
            style_dim=style_dim,
            upsample=False,
            blur_kernel=blur_kernel
        )
        self.to_rgb_1 = ToRGB(
            self.channels[start_resolution],
            style_dim=style_dim,
            upsample=False
        )
        in_channels = self.channels[start_resolution]

        for layer_idx in range(self.num_layers):
            # default (layer_idx + 5) // 2
            res = (layer_idx + 2 * start_res_log + 1) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(start_res_log + 1, resol_log + 1):
            out_channels = self.channels[2 ** i]

            self.to_rgbs.append(
                ToRGB(
                    in_channels=out_channels,
                    style_dim=style_dim
                )
            )

            self.convs.append(
                StyledConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    style_dim=style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel
                )
            )

            self.convs.append(
                StyledConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    style_dim=style_dim,
                    upsample=False,
                    blur_kernel=blur_kernel
                )
            )

            in_channels = out_channels
        pass

    def create_noise(self):
        device = self.input.output.device
        noises = [torch.randn(1, 1, self.start_resol, self.start_resol, device=device)]

        for i in range(self.start_res_log + 1, self.resol_log + 1):
            for j in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    def mean_latents(self, n_latent):
        latents_in = torch.randn(
            n_latent, self.style_dim, device=self.input.output.device
        )
        latents = self.styler(latents_in).mean(0, keepdim=True)
        return latents

    def get_latent(self, x):
        return self.styler(x)

    def __nf(self, layer, fmap_base, fmap_decay, fmap_min, fmap_max):
        return np.clip(int(fmap_base / (2.0 ** (layer * fmap_decay))), fmap_min, fmap_max)

    def forward(self,
                styles,
                return_latents=False,
                truncation=None,
                truncate_latent=None,
                inject_index=False,
                is_input_latent=False,
                noise=None,
                randomize_noise=True
                ):
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if not is_input_latent:
            styles = [self.styler(style) for style in styles]

        if truncation < 1:
            styles = [truncate_latent + truncation * (style - truncate_latent) for style in styles]

        if len(styles) < 2:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent_2 = styles[1].unsqueeze(1).repear(1, self.num_latent - inject_index, 1)

            latent = torch.cat([latent, latent_2], 1)

        out = self.input(latent)
        out = self.conv_1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb_1(out, latent[:, 1])

        i = 1
        for conv0, conv1, noise0, noise1, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv0(out, latent[:, i], noise=noise0)
            out = conv1(out, latent[:, i + 1], noise=noise1)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2
        del i

        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None
