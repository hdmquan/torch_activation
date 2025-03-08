import torch
import torch.nn as nn
from torch import Tensor

from torch_activation import register_activation


@register_activation
class SQRT(nn.Module):
    r"""
    Applies the Square-root-based activation function (SQRT):

    :math:`\text{SQRT}(z) = \begin{cases} 
    \sqrt{z}, & z \geq 0 \\
    -\sqrt{-z}, & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    References:
        - Noel et al. "Square-root-based activation functions for deep learning." (2021)
    """

    def __init__(self, inplace: bool = False):
        super(SQRT, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        if self.inplace:
            return z.where(z >= 0, torch.sqrt(z), -torch.sqrt(-z))
        else:
            return torch.where(z >= 0, torch.sqrt(z), -torch.sqrt(-z))


@register_activation
class SSAF(nn.Module):
    r"""
    Applies the S-shaped activation function (SSAF), a parametric variant of SQRT:

    :math:`\text{SSAF}(z) = \begin{cases} 
    \sqrt{2az}, & z \geq 0 \\
    -\sqrt{-2az}, & z < 0 
    \end{cases}`

    where :math:`a` is a fixed parameter.

    Args:
        a (float, optional): The scaling parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    References:
        - Proposed independently as "S-shaped activation function" (SSAF)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(SSAF, self).__init__()
        self.a = a
        self.factor = 2 * a
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        if self.inplace:
            return z.where(z >= 0, torch.sqrt(self.factor * z), -torch.sqrt(-self.factor * z))
        else:
            return torch.where(z >= 0, torch.sqrt(self.factor * z), -torch.sqrt(-self.factor * z))
