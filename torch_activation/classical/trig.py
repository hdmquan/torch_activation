import torch
import torch.nn as nn
from torch import Tensor
import math

from torch_activation import register_activation


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


@register_activation
class CombHSine(nn.Module):
    r"""
    Applies the Comb-H-sine activation function:

    :math:`\text{CombHSine}(z) = \sinh(az) + \sinh^{-1}(az)`

    where sinh(x) is the hyperbolic sine, sinh^{-1}(x) is its inverse,
    and a is a predefined hyperparameter.

    Args:
        a (float, optional): hyperparameter controlling the scaling. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(CombHSine, self).__init__()
        self.a = a
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        az = self.a * z
        return torch.sinh(az) + torch.asinh(az)


@register_activation
class ModifiedArcsinh(nn.Module):
    r"""
    Applies the Modified arcsinh (m-arcsinh) activation function:

    :math:`\text{ModifiedArcsinh}(z) = \frac{1}{12} \sinh^{-1}(z) |z|`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(ModifiedArcsinh, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return (1.0 / 12.0) * torch.asinh(z) * torch.abs(z)


@register_activation
class HyperSinh(nn.Module):
    r"""
    Applies the hyper-sinh activation function:

    :math:`\text{HyperSinh}(z) = 
    \begin{cases} 
    \frac{\sinh(z)}{3}, & \text{if } z > 0 \\
    \frac{z^3}{4}, & \text{if } z \leq 0
    \end{cases}`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(HyperSinh, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        positive_part = torch.sinh(z) / 3.0
        negative_part = torch.pow(z, 3) / 4.0
        return torch.where(z > 0, positive_part, negative_part)


@register_activation
class Arctid(nn.Module):
    r"""
    Applies the Arctid activation function:

    :math:`\text{Arctid}(z) = \tan^{-1}(z) \cdot 2^{-z}`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Arctid, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return torch.atan(z) * torch.pow(2.0, -z)


@register_activation
class Cosine(nn.Module):
    r"""
    Applies the Cosine activation function:

    :math:`\text{Cosine}(z) = 1 - \cos(z)`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Cosine, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return 1.0 - torch.cos(z)


@register_activation
class Cosid(nn.Module):
    r"""
    Applies the Cosid activation function:

    :math:`\text{Cosid}(z) = \cos(z) - z`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Cosid, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return torch.cos(z) - z


@register_activation
class Sinp(nn.Module):
    r"""
    Applies the Sinp activation function:

    :math:`\text{Sinp}(z) = \sin(z) - az`

    where a is a fixed parameter.

    Args:
        a (float, optional): scaling parameter for the linear term. Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(Sinp, self).__init__()
        self.a = a
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return torch.sin(z) - self.a * z


@register_activation
class GCU(nn.Module):
    r"""
    Applies the Growing Cosine Unit (GCU) activation function:

    :math:`\text{GCU}(z) = z \cdot \cos(z)`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(GCU, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return z * torch.cos(z)


@register_activation
class ASU(nn.Module):
    r"""
    Applies the Amplifying Sine Unit (ASU) activation function:

    :math:`\text{ASU}(z) = z \cdot \sin(z)`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(ASU, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return z * torch.sin(z)


@register_activation
class Sinc(nn.Module):
    r"""
    Applies the Sinc activation function:

    :math:`\text{Sinc}(z) = 
    \begin{cases} 
    \frac{\sin(\pi z)}{\pi z}, & \text{if } z \neq 0 \\
    1, & \text{if } z = 0
    \end{cases}`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Sinc, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        # Handle the case where z = 0 to avoid division by zero
        return torch.where(z == 0, torch.ones_like(z), torch.sin(math.pi * z) / (math.pi * z))


@register_activation
class SSU(nn.Module):
    r"""
    Applies the Shifted Sine Unit (SSU) activation function:

    :math:`\text{SSU}(z) = \pi \cdot \text{sinc}(z - \pi)`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(SSU, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        shifted_z = z - math.pi
        # Handle the case where shifted_z = 0 to avoid division by zero
        return math.pi * torch.where(
            shifted_z == 0, 
            torch.ones_like(z), 
            torch.sin(math.pi * shifted_z) / (math.pi * shifted_z)
        )


@register_activation
class DSU(nn.Module):
    r"""
    Applies the Decaying Sine Unit (DSU) activation function:

    :math:`\text{DSU}(z) = \frac{\pi}{2} \cdot (\text{sinc}(z - \pi) - \text{sinc}(z + \pi))`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(DSU, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        left_shifted = z - math.pi
        right_shifted = z + math.pi
        
        # Handle the case where shifted values are 0 to avoid division by zero
        left_sinc = torch.where(
            left_shifted == 0, 
            torch.ones_like(z), 
            torch.sin(math.pi * left_shifted) / (math.pi * left_shifted)
        )
        
        right_sinc = torch.where(
            right_shifted == 0, 
            torch.ones_like(z), 
            torch.sin(math.pi * right_shifted) / (math.pi * right_shifted)
        )
        
        return (math.pi / 2.0) * (left_sinc - right_sinc)


@register_activation
class HcLSH(nn.Module):
    r"""
    Applies the Hyperbolic Cosine Linearized Squashing Function (HcLSH) activation function:

    :math:`\text{HcLSH}(z) = 
    \begin{cases} 
    \ln(\cosh(z)) + \frac{z \cdot \cosh(z)}{2}, & \text{if } z \geq 0 \\
    \ln(\cosh(z)) + z, & \text{if } z < 0
    \end{cases}`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(HcLSH, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        log_cosh = torch.log(torch.cosh(z))
        positive_part = log_cosh + (z * torch.cosh(z)) / 2.0
        negative_part = log_cosh + z
        return torch.where(z >= 0, positive_part, negative_part)