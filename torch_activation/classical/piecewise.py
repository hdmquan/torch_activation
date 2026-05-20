import math

import torch
import torch.nn as nn
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class BiFiring(BaseActivation):
    r"""
    Applies the Bi-Firing activation function (bfire):

    :math:`\text{BiFiring}(z) = \begin{cases} 
    z - \frac{a}{2}, & z > a \\
    \frac{z^2}{2a}, & -a \leq z \leq a \\
    -z - \frac{a}{2}, & z < -a 
    \end{cases}`

    A smoothed variant of vReLU that becomes vReLU as a→0.

    Args:
        a (float, optional): smoothing hyperparameter. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
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
        result = torch.zeros_like(z)

        # z > a
        upper_mask = z > self.a
        result[upper_mask] = z[upper_mask] - self.a / 2

        # -a <= z <= a
        mid_mask = (z >= -self.a) & (z <= self.a)
        result[mid_mask] = (z[mid_mask] ** 2) / (2 * self.a)

        # z < -a
        lower_mask = z < -self.a
        result[lower_mask] = -z[lower_mask] - self.a / 2

        return result


@register_activation
class BoundedBiFiring(BaseActivation):
    r"""
    Applies the Bounded Bi-Firing activation function (bbfire):

    :math:`\text{BoundedBiFiring}(z) = \begin{cases} 
    b, & z < -b - \frac{a}{2} \\
    -z - \frac{a}{2}, & -b - \frac{a}{2} \leq z < -a \\
    \frac{z^2}{2a}, & -a \leq z \leq a \\
    z - \frac{a}{2}, & a < z \leq b + \frac{a}{2} \\
    b, & z > b + \frac{a}{2}
    \end{cases}`

    A bounded variant of the bi-firing activation function that is symmetrical about the origin
    and has a near inverse-bell-shaped activation curve.

    Args:
        a (float, optional): smoothing hyperparameter. Default: ``1.0``
        b (float, optional): bounding hyperparameter. Default: ``5.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, b: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        # Unused

    def _forward(self, z) -> Tensor:
        result = torch.zeros_like(z)

        # z < -b - a/2
        lower_bound_mask = z < (-self.b - self.a / 2)
        result[lower_bound_mask] = self.b

        # -b - a/2 <= z < -a
        lower_mid_mask = (z >= (-self.b - self.a / 2)) & (z < -self.a)
        result[lower_mid_mask] = -z[lower_mid_mask] - self.a / 2

        # -a <= z <= a
        mid_mask = (z >= -self.a) & (z <= self.a)
        result[mid_mask] = (z[mid_mask] ** 2) / (2 * self.a)

        # a < z <= b + a/2
        upper_mid_mask = (z > self.a) & (z <= (self.b + self.a / 2))
        result[upper_mid_mask] = z[upper_mid_mask] - self.a / 2

        # z > b + a/2
        upper_bound_mask = z > (self.b + self.a / 2)
        result[upper_bound_mask] = self.b

        return result


@register_activation
class PiecewiseMexicanHat(BaseActivation):
    r"""
    Applies the Piecewise Mexican-Hat activation function (PMAF):

    :math:`\text{PMAF}(z) = \begin{cases} 
    \frac{1}{\sqrt{3}\pi} - \frac{1}{4}(1-(z+a)^2) \exp(-\frac{(z+a)^2}{2}), & z < 0 \\
    \frac{1}{\sqrt{3}\pi} - \frac{1}{4}(1-(z-a)^2) \exp(-\frac{(z-a)^2}{2}), & z \geq 0
    \end{cases}`

    Args:
        a (float, optional): shape parameter. Default: ``4.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        # Unused
        self.const_term = 1 / (math.sqrt(3) * math.pi)

    def _forward(self, z) -> Tensor:
        shifted_neg = z + self.a
        val_neg = self.const_term - 0.25 * (1 - shifted_neg**2) * torch.exp(-(shifted_neg**2) / 2)
        shifted_pos = z - self.a
        val_pos = self.const_term - 0.25 * (1 - shifted_pos**2) * torch.exp(-(shifted_pos**2) / 2)
        return torch.where(z < 0, val_neg, val_pos)


@register_activation
class PiecewiseRadialBasisFunction(BaseActivation):
    r"""
    Applies the Piecewise Radial Basis Function (PRBF):

    :math:`\text{PRBF}(z) = \begin{cases} 
    \exp(-\frac{(z-2a)^2}{b^2}), & z \geq a \\
    \exp(-\frac{z^2}{b^2}), & -a < z < a \\
    \exp(-\frac{(z+2a)^2}{b^2}), & z \leq -a
    \end{cases}`

    Args:
        a (float, optional): shape parameter. Default: ``3.0``
        b (float, optional): scale parameter. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 3.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        # Unused

    def _forward(self, z) -> Tensor:
        val_upper = torch.exp(-((z - 2 * self.a) ** 2) / (self.b**2))
        val_mid = torch.exp(-(z**2) / (self.b**2))
        val_lower = torch.exp(-((z + 2 * self.a) ** 2) / (self.b**2))
        return torch.where(z >= self.a, val_upper, torch.where(z <= -self.a, val_lower, val_mid))
