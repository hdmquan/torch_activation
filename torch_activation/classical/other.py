import torch
import torch.nn as nn
from torch import Tensor
import math

from torch_activation import register_activation

@register_activation
class Binary(nn.Module):
    r"""
    Applies the Binary activation function:

    :math:`\text{Binary}(z) = \begin{cases} 
    0, & z < 0 \\
    1, & z \geq 0 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Binary, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return _Binary.apply(z)


class _Binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # Pass the gradient through unchanged
        return grad_output


@register_activation
class Sine(nn.Module):
    r"""
    Applies the Sine activation function:

    :math:`\text{Sine}(z) = \sin(\pi \cdot z)`

    Args:
        omega (float, optional): frequency of the sine wave. Default: ``math.pi``
        inplace (bool, optional): parameter kept for API consistency, but sine operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, omega: float = math.pi, inplace: bool = False):
        super(Sine, self).__init__()
        self.omega = omega
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return torch.sin(self.omega * z)