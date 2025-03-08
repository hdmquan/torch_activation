import torch
import torch.nn as nn
from torch import Tensor
import math

from torch_activation import register_activation


@register_activation
class Softplus(nn.Module):
    r"""
    Applies the Softplus activation function:

    :math:`\text{Softplus}(z) = \ln(\exp(z) + 1)`

    The softplus is a smooth approximation to the ReLU function and
    has a non-zero gradient for negative inputs.

    Args:
        beta (float, optional): controls the smoothness of the approximation. Default: ``1.0``
        threshold (float, optional): values above this revert to a linear function. Default: ``20.0``
        inplace (bool, optional): parameter kept for API consistency, but softplus operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0, inplace: bool = False):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        # Use the built-in softplus for numerical stability
        return torch.nn.functional.softplus(z, self.beta, self.threshold)


@register_activation
class ParametricSoftplus(nn.Module):
    r"""
    Applies the Parametric Softplus (PSoftplus) activation function:

    :math:`\text{PSoftplus}(z) = a \cdot (\ln(\exp(z) + 1) - b)`

    Where a and b are fixed predetermined hyperparameters. The recommended values
    are a=1.5 and b=ln(2).

    Args:
        a (float, optional): scaling parameter. Default: ``1.5``
        b (float, optional): shifting parameter. Default: ``0.693`` (ln(2))
        beta (float, optional): controls the smoothness of the approximation. Default: ``1.0``
        threshold (float, optional): values above this revert to a linear function. Default: ``20.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.5, b: float = 0.693, beta: float = 1.0, 
                 threshold: float = 20.0, inplace: bool = False):
        super(ParametricSoftplus, self).__init__()
        self.a = a
        self.b = b
        self.beta = beta
        self.threshold = threshold
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        softplus = torch.nn.functional.softplus(z, self.beta, self.threshold)
        return self.a * (softplus - self.b)


@register_activation
class SoftPlusPlus(nn.Module):
    r"""
    Applies the Soft++ activation function:

    :math:`\text{Soft++}(z) = \ln(1 + \exp(a \cdot z)) + \frac{z}{b} - \ln(2)`

    Where a and b are fixed predetermined hyperparameters. The recommended values
    are a=1 and b=2.

    Args:
        a (float, optional): scaling parameter for the input in softplus term. Default: ``1.0``
        b (float, optional): scaling parameter for the linear term. Default: ``2.0``
        threshold (float, optional): values above this revert to a linear function. Default: ``20.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, threshold: float = 20.0, inplace: bool = False):
        super(SoftPlusPlus, self).__init__()
        self.a = a
        self.b = b
        self.threshold = threshold
        self.ln2 = math.log(2)
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        # Apply a to the input for the softplus term
        scaled_input = self.a * z
        
        # For numerical stability, use the built-in softplus for the first term
        softplus_term = torch.nn.functional.softplus(scaled_input, beta=1.0, threshold=self.threshold)
        
        # Add the linear term and subtract ln(2)
        return softplus_term + (z / self.b) - self.ln2


@register_activation
class RandSoftplus(nn.Module):
    r"""
    Applies the Rand Softplus (RSP) activation function:

    :math:`\text{RSP}(z) = (1 - a) \cdot \max(0, z) + a \cdot \ln(1 + \exp(z))`

    Where a is a stochastic parameter determined by the noise level of the input data.
    In this implementation, a is a fixed parameter for simplicity.

    Args:
        a (float, optional): interpolation parameter between ReLU and softplus. Default: ``0.5``
        beta (float, optional): controls the smoothness of the softplus. Default: ``1.0``
        threshold (float, optional): values above this revert to a linear function. Default: ``20.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 0.5, beta: float = 1.0, threshold: float = 20.0, inplace: bool = False):
        super(RandSoftplus, self).__init__()
        self.a = a
        self.beta = beta
        self.threshold = threshold
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        # ReLU term
        relu_term = torch.nn.functional.relu(z)
        
        # Softplus term
        softplus_term = torch.nn.functional.softplus(z, self.beta, self.threshold)
        
        # Interpolate between ReLU and softplus based on parameter a
        return (1 - self.a) * relu_term + self.a * softplus_term
