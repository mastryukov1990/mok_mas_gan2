import os

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.nn import functional as F

module_path = os.path.dirname(__file__)
fused = load("fused",
             sources=[
                 os.path.join(module_path, "fused_bias_act.cpp"),
                 os.path.join(module_path, "fused_bias_act_kernel.cu"),
             ],
             )


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
<<<<<<< HEAD
    def forward(ctx, grad_output, out, negative_slope, scale):
=======
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
>>>>>>> main
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]
        if grad_input.dim > 2:
            dim += list(range(2, grad_input.dim))

<<<<<<< HEAD
        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    def backward(ctx, grad_input_, grad_bias_):
        out, = ctx.saved_tensors
        grad_out_ = fused.fused_bias_act(
            grad_input_, grad_bias_, out, 3, 1, ctx.negative_slope, ctx.scale
        )
        return grad_out_, None, None, None
=======
        if bias:
            grad_bias = grad_input.sum(dim).detach()
        else:
            grad_bias = None

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )
        return gradgrad_out, None, None, None, None
>>>>>>> main


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
<<<<<<< HEAD
=======
        if bias is None:
            bias = empty

        ctx.bias = bias is not None

>>>>>>> main
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
<<<<<<< HEAD
            grad_output, out, ctx.negative_slope, ctx.scale
=======
            grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale
>>>>>>> main
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
<<<<<<< HEAD
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
=======
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None

>>>>>>> main
        self.negative_slope = negative_slope
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(channel))

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == "cpu":
<<<<<<< HEAD
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
                F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
                             ) * scale
        )

    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)

=======
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return (F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
                                 ) * scale
                    )
        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale

    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
>>>>>>> main
