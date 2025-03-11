import torch
import torch.nn as nn
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class Polyexp(BaseActivation):
    r"""
    Applies the Polyexp activation function:

    :math:`\text{Polyexp}(z) = a \cdot z^2 + b \cdot z + c \cdot \exp(-d \cdot z^2)`

    An activation function combining quadratic function and an exponential function.

    Args:
        a (float, optional): Parameter for the quadratic term. Default: ``1.0``
        b (float, optional): Parameter for the linear term. Default: ``1.0``
        c (float, optional): Parameter for the exponential term. Default: ``1.0``
        d (float, optional): Parameter for the exponential decay. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but polyexp operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
          # Unused

    def _forward(self, z) -> Tensor:
        return self.a * z**2 + self.b * z + self.c * torch.exp(-self.d * z**2)


@register_activation
class Exponential(BaseActivation):
    r"""
    Applies the Exponential activation function:

    :math:`\text{Exponential}(z) = \exp(-z)`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but exponential operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
          # Unused

    def _forward(self, z) -> Tensor:
        return torch.exp(-z)

@register_activation
class Symexp(BaseActivation):
    r"""
    Applies the Symexp activation function:

    :math:`\text{Symexp}(z) = \text{sgn}(z) \cdot (\exp(|z|) - 1)`

    Inverse of the logmoid activation unit (LAU).

    Args:
        inplace (bool, optional): parameter kept for API consistency, but symexp operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
          # Unused

    def _forward(self, z) -> Tensor:
        return torch.sign(z) * (torch.exp(torch.abs(z)) - 1)

@register_activation
class Wave(BaseActivation):
    r"""
    Applies the Wave activation function:

    :math:`\text{Wave}(z) = 1 - z^2 \cdot \exp(-a \cdot z^2)`

    An activation function combining quadratic function and an exponential function.

    Args:
        a (float, optional): Parameter for the exponential decay. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but wave operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
          # Unused

    def _forward(self, z) -> Tensor:
        return 1 - z**2 * torch.exp(-self.a * z**2)