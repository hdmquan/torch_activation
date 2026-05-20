import torch
import torch.nn as nn
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class SQRT(BaseActivation):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, z) -> Tensor:
        pos = z.clamp(min=0)
        neg = (-z).clamp(min=0)
        return torch.where(z >= 0, torch.sqrt(pos), -torch.sqrt(neg))


@register_activation
class SSAF(BaseActivation):
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

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.factor = 2 * a

    def _forward(self, z) -> Tensor:
        pos = (self.factor * z).clamp(min=0)
        neg = (-self.factor * z).clamp(min=0)
        return torch.where(z >= 0, torch.sqrt(pos), -torch.sqrt(neg))
