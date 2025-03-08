import torch
import torch.nn as nn
from torch import Tensor
import math

from torch_activation import register_activation


@register_activation
class BiFiring(nn.Module):
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

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(BiFiring, self).__init__()
        self.a = a
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        result = torch.zeros_like(z)
        
        # z > a
        upper_mask = z > self.a
        result[upper_mask] = z[upper_mask] - self.a/2
        
        # -a <= z <= a
        mid_mask = (z >= -self.a) & (z <= self.a)
        result[mid_mask] = (z[mid_mask]**2) / (2 * self.a)
        
        # z < -a
        lower_mask = z < -self.a
        result[lower_mask] = -z[lower_mask] - self.a/2
        
        return result


@register_activation
class BoundedBiFiring(nn.Module):
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

    def __init__(self, a: float = 1.0, b: float = 5.0, inplace: bool = False):
        super(BoundedBiFiring, self).__init__()
        self.a = a
        self.b = b
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        result = torch.zeros_like(z)
        
        # z < -b - a/2
        lower_bound_mask = z < (-self.b - self.a/2)
        result[lower_bound_mask] = -self.b
        
        # -b - a/2 <= z < -a
        lower_mid_mask = (z >= (-self.b - self.a/2)) & (z < -self.a)
        result[lower_mid_mask] = -z[lower_mid_mask] - self.a/2
        
        # -a <= z <= a
        mid_mask = (z >= -self.a) & (z <= self.a)
        result[mid_mask] = (z[mid_mask]**2) / (2 * self.a)
        
        # a < z <= b + a/2
        upper_mid_mask = (z > self.a) & (z <= (self.b + self.a/2))
        result[upper_mid_mask] = z[upper_mid_mask] - self.a/2
        
        # z > b + a/2
        upper_bound_mask = z > (self.b + self.a/2)
        result[upper_bound_mask] = self.b
        
        return result


@register_activation
class PiecewiseMexicanHat(nn.Module):
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

    def __init__(self, a: float = 4.0, inplace: bool = False):
        super(PiecewiseMexicanHat, self).__init__()
        self.a = a
        self.inplace = inplace  # Unused
        self.const_term = 1 / (math.sqrt(3) * math.pi)

    def forward(self, z) -> Tensor:
        result = torch.zeros_like(z)
        
        # z < 0
        neg_mask = z < 0
        z_neg = z[neg_mask]
        shifted_z_neg = z_neg + self.a
        exp_term_neg = torch.exp(-(shifted_z_neg**2) / 2)
        result[neg_mask] = self.const_term - 0.25 * (1 - shifted_z_neg**2) * exp_term_neg
        
        # z >= 0
        pos_mask = z >= 0
        z_pos = z[pos_mask]
        shifted_z_pos = z_pos - self.a
        exp_term_pos = torch.exp(-(shifted_z_pos**2) / 2)
        result[pos_mask] = self.const_term - 0.25 * (1 - shifted_z_pos**2) * exp_term_pos
        
        return result


@register_activation
class PiecewiseRadialBasisFunction(nn.Module):
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

    def __init__(self, a: float = 3.0, b: float = 1.0, inplace: bool = False):
        super(PiecewiseRadialBasisFunction, self).__init__()
        self.a = a
        self.b = b
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        result = torch.zeros_like(z)
        
        # z >= a
        upper_mask = z >= self.a
        z_upper = z[upper_mask]
        result[upper_mask] = torch.exp(-((z_upper - 2*self.a)**2) / (self.b**2))
        
        # -a < z < a
        mid_mask = (z > -self.a) & (z < self.a)
        z_mid = z[mid_mask]
        result[mid_mask] = torch.exp(-(z_mid**2) / (self.b**2))
        
        # z <= -a
        lower_mask = z <= -self.a
        z_lower = z[lower_mask]
        result[lower_mask] = torch.exp(-((z_lower + 2*self.a)**2) / (self.b**2))
        
        return result
