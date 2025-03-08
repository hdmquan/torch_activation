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
class BentIdentity(nn.Module):
    r"""
    Applies the Bent Identity activation function:

    :math:`\text{BentIdentity}(z) = \frac{\sqrt{z^2 + 1} - 1}{2} + z`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but bent identity operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(BentIdentity, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return (torch.sqrt(z**2 + 1) - 1) / 2 + z


@register_activation
class Mishra(nn.Module):
    r"""
    Applies the Mishra activation function:

    :math:`\text{Mishra}(z) = \frac{1}{2} \cdot \frac{z}{1 + |z|} + \frac{z}{2} \cdot \frac{1}{1 + |z|}`

    Args:
        inplace (bool, optional): parameter kept for API consistency, but Mishra operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Mishra, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        abs_z = torch.abs(z)
        term1 = 0.5 * z / (1 + abs_z)
        term2 = 0.5 * z / (1 + abs_z)
        return term1 + term2


@register_activation
class SahaBora(nn.Module):
    r"""
    Applies the Saha-Bora activation function (SBAF):

    :math:`\text{SahaBora}(z) = \frac{1}{1 + k \cdot z^{\alpha} \cdot (1-z)^{(1-\alpha)}}`

    Args:
        k (float, optional): non-trainable parameter. Default: ``0.98``
        alpha (float, optional): non-trainable parameter. Default: ``0.5``
        inplace (bool, optional): parameter kept for API consistency, but SBAF operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, k: float = 0.98, alpha: float = 0.5, inplace: bool = False):
        super(SahaBora, self).__init__()
        self.k = k
        self.alpha = alpha
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        # Clamp z to avoid numerical issues when z is close to 0 or 1
        z_safe = torch.clamp(z, min=1e-7, max=1-1e-7)
        denominator = 1 + self.k * (z_safe**self.alpha) * ((1 - z_safe)**(1 - self.alpha))
        return 1 / denominator


@register_activation
class Logarithmic(nn.Module):
    r"""
    Applies the Logarithmic activation function (LAF):

    :math:`\text{Logarithmic}(z) = \begin{cases} 
    \ln(z) + 1, & z \geq 0 \\
    -\ln(-z) + 1, & z < 0 
    \end{cases}`

    Also known as symlog in some literature.

    Args:
        inplace (bool, optional): parameter kept for API consistency, but logarithmic operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False):
        super(Logarithmic, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        # Add small epsilon to avoid log(0)
        eps = 1e-7
        pos_mask = z >= 0
        result = torch.zeros_like(z)
        result[pos_mask] = torch.log(z[pos_mask] + eps) + 1
        result[~pos_mask] = -torch.log(-z[~pos_mask] + eps) + 1
        return result


@register_activation
class Symexp(nn.Module):
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

    def __init__(self, inplace: bool = False):
        super(Symexp, self).__init__()
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return torch.sign(z) * (torch.exp(torch.abs(z)) - 1)


@register_activation
class SPOCU(nn.Module):
    r"""
    Applies the Scaled Polynomial Constant Unit (SPOCU) activation function:

    :math:`\text{SPOCU}(z) = a \cdot h(z)^c + b - a \cdot h(b)`

    where:
    
    :math:`h(x) = \begin{cases} 
    r(d), & x \geq d \\
    r(x), & 0 \leq x < d \\
    x, & x < 0 
    \end{cases}`
    
    and :math:`r(x) = x^3 - \frac{2x^4 + x^5}{2}`

    Args:
        a (float, optional): scaling parameter. Default: ``1.0``
        b (float, optional): parameter in range (0,1). Default: ``0.5``
        c (float, optional): exponent parameter. Default: ``1.0``
        d (float, optional): threshold parameter in range [1,∞). Default: ``1.0``
        inplace (bool, optional): parameter kept for API consistency, but SPOCU operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, b: float = 0.5, c: float = 1.0, d: float = 1.0, inplace: bool = False):
        super(SPOCU, self).__init__()
        assert a > 0, "Parameter a must be positive"
        assert 0 < b < 1, "Parameter b must be in range (0,1)"
        assert c > 0, "Parameter c must be positive"
        assert d >= 1, "Parameter d must be >= 1"
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.inplace = inplace  # Unused
        
        # Pre-compute h(b) for efficiency
        self.h_b = self._r(b) if 0 <= b < d else self._r(d)

    def _r(self, x):
        return x**3 - (2*x**4 + x**5)/2

    def _h(self, x):
        neg_mask = x < 0
        mid_mask = (0 <= x) & (x < self.d)
        high_mask = x >= self.d
        
        result = torch.zeros_like(x)
        result[neg_mask] = x[neg_mask]
        result[mid_mask] = self._r(x[mid_mask])
        result[high_mask] = self._r(torch.tensor(self.d, device=x.device))
        
        return result

    def forward(self, z) -> Tensor:
        h_z = self._h(z)
        return self.a * (h_z**self.c) + self.b - self.a * self.h_b


@register_activation
class PUAF(nn.Module):
    r"""
    Applies the Polynomial Universal Activation Function (PUAF):

    :math:`\text{PUAF}(z) = \begin{cases} 
    z^a, & z > c \\
    z^a \cdot \frac{(c+z)^b}{(c+z)^b+(c-z)^b}, & |z| \leq c \\
    0, & z < -c 
    \end{cases}`

    Can approximate various activation functions based on parameter settings:
    - ReLU: a=1, b=0, c=0
    - Logistic sigmoid (approx): a=0, b=5, c=10
    - Swish (approx): a=1, b=5, c=10

    Args:
        a (float, optional): exponent parameter. Default: ``1.0``
        b (float, optional): exponent parameter. Default: ``5.0``
        c (float, optional): threshold parameter. Default: ``10.0``
        inplace (bool, optional): parameter kept for API consistency, but PUAF operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 1.0, b: float = 5.0, c: float = 10.0, inplace: bool = False):
        super(PUAF, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        result = torch.zeros_like(z)
        
        # z > c
        upper_mask = z > self.c
        result[upper_mask] = z[upper_mask] ** self.a
        
        # |z| <= c
        mid_mask = torch.abs(z) <= self.c
        if self.b == 0:
            # Handle special case to avoid division by zero
            result[mid_mask] = z[mid_mask] ** self.a
        else:
            z_mid = z[mid_mask]
            numerator = (self.c + z_mid) ** self.b
            denominator = numerator + (self.c - z_mid) ** self.b
            result[mid_mask] = (z_mid ** self.a) * (numerator / denominator)
        
        # z < -c is already set to 0 by initialization
        
        return result


@register_activation
class ArandaOrdaz(nn.Module):
    r"""
    Applies the Aranda-Ordaz activation function:

    :math:`\text{ArandaOrdaz}(z) = 1 - (1 + a \cdot \exp(z))^{-1}`

    Args:
        a (float, optional): fixed parameter. Default: ``2.0``
        inplace (bool, optional): parameter kept for API consistency, but Aranda-Ordaz operation 
                                 cannot be done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, a: float = 2.0, inplace: bool = False):
        super(ArandaOrdaz, self).__init__()
        assert a > 0, "Parameter a must be positive"
        self.a = a
        self.inplace = inplace  # Unused

    def forward(self, z) -> Tensor:
        return 1 - (1 + self.a * torch.exp(z))**(-1)