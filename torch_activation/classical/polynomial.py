import torch
import torch.nn as nn
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class NCU(BaseActivation):
    r"""
    Applies the Non-monotonic Cubic Unit (NCU) activation function:

    :math:`\text{NCU}(z) = z - z^3`

    A simple activation function based on a third-degree polynomial.

    Args:
        inplace (bool, optional): parameter kept for API consistency, but NCU operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
          # Unused

    def _forward(self, z) -> Tensor:
        return z - z**3


@register_activation
class Triple(BaseActivation):
    r"""
    Applies the Triple activation function:

    :math:`\text{Triple}(z) = a \cdot z^3`

    An activation function based on a third-degree polynomial.

    Args:
        a (float, optional): Parameter for the cubic term. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but triple operation 
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
        return self.a * z**3


@register_activation
class SQU(BaseActivation):
    r"""
    Applies the Shifted Quadratic Unit (SQU) activation function:

    :math:`\text{SQU}(z) = z^2 + z`

    A simple non-monotonic activation function.

    Args:
        inplace (bool, optional): parameter kept for API consistency, but SQU operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
          # Unused

    def _forward(self, z) -> Tensor:
        return z**2 + z
