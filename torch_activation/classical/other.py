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
        if self.inplace:
            z.where(z < 0, torch.zeros_like(z), torch.ones_like(z))
            return z
        else:
            return torch.where(z < 0, torch.zeros_like(z), torch.ones_like(z))
    

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