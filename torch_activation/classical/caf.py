import torch
import torch.nn as nn
from torch import Tensor
from torch_activation import register_activation

class HCAF(nn.Module):
    r"""
    Applies the Hybrid Chaotic Activation Function:

    .. math::
        a_i = \sigma(z_i)
        
        c_{i,1} = ra_i(1 - a_i)
        
        c_{i,j} = rc_{i,j-1}(1 - c_{i,j-1})

    where :math:`\sigma(z_i)` is the logistic sigmoid and :math:`r = 4` by default.

    Args:
        r (float, optional): Chaotic parameter. Default: ``4.0``
        iterations (int, optional): Number of iterations for the chaotic map. Default: ``3``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.HCAF()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HCAF(r=3.9, iterations=5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, r: float = 4.0, iterations: int = 3):
        super().__init__()
        self.r = r
        self.iterations = iterations

    def forward(self, x: Tensor) -> Tensor:
        # Initial sigmoid activation
        a = torch.sigmoid(x)
        
        # First chaotic iteration
        c = self.r * a * (1 - a)
        
        # Additional chaotic iterations
        for _ in range(1, self.iterations):
            c = self.r * c * (1 - c)
            
        return c


@register_activation
class FCAF_Hidden(nn.Module):
    r"""
    Applies the Fusion of Chaotic Activation Function for hidden units:

    .. math::
        f(z_{i+1}) = rz_i(1 - z_i) + z_i + a - \frac{b}{2\pi} \sin(2\pi z_i)

    Args:
        r (float, optional): Chaotic parameter. Default: ``4.0``
        a (float, optional): Linear shift parameter. Default: ``0.0``
        b (float, optional): Sinusoidal amplitude parameter. Default: ``0.5``
        iterations (int, optional): Number of iterations for the chaotic map. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.FCAF_Hidden()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.FCAF_Hidden(r=3.9, a=0.1, b=0.3, iterations=2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, r: float = 4.0, a: float = 0.0, b: float = 0.5, iterations: int = 1):
        super().__init__()
        self.r = r
        self.a = a
        self.b = b
        self.iterations = iterations

    def forward(self, x: Tensor) -> Tensor:
        # Normalize input to [0,1] range for chaotic map stability
        z = torch.sigmoid(x)
        
        for _ in range(self.iterations):
            # Apply the chaotic map
            z = self.r * z * (1 - z) + z + self.a - (self.b / (2 * torch.pi)) * torch.sin(2 * torch.pi * z)
            
        return z


@register_activation
class FCAF_Output(nn.Module):
    r"""
    Applies the Fusion of Chaotic Activation Function for output units:

    .. math::
        f(z_{i+1}) = rz_i(1 - z_i) + z_i + a - \frac{b}{2\pi} \sin(2\pi z_i) + \exp(-cz_i^2) + d

    Args:
        r (float, optional): Chaotic parameter. Default: ``4.0``
        a (float, optional): Linear shift parameter. Default: ``0.0``
        b (float, optional): Sinusoidal amplitude parameter. Default: ``0.5``
        c (float, optional): Gaussian width parameter. Default: ``1.0``
        d (float, optional): Constant shift parameter. Default: ``0.0``
        iterations (int, optional): Number of iterations for the chaotic map. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.FCAF_Output()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.FCAF_Output(r=3.9, a=0.1, b=0.3, c=2.0, d=0.1, iterations=2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, r: float = 4.0, a: float = 0.0, b: float = 0.5, 
                 c: float = 1.0, d: float = 0.0, iterations: int = 1):
        super().__init__()
        self.r = r
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.iterations = iterations

    def forward(self, x: Tensor) -> Tensor:
        # Normalize input to [0,1] range for chaotic map stability
        z = torch.sigmoid(x)
        
        for _ in range(self.iterations):
            # Apply the chaotic map with additional terms
            z = (self.r * z * (1 - z) + 
                 z + self.a - 
                 (self.b / (2 * torch.pi)) * torch.sin(2 * torch.pi * z) + 
                 torch.exp(-self.c * z * z) + 
                 self.d)
            
        return z


@register_activation
class CCAF(nn.Module):
    r"""
    Applies the Cascade Chaotic Activation Function:

    .. math::
        f(z_{i+1}) = a \cdot \sin(\pi \cdot b \cdot \sin(\pi z_i))

    where :math:`a, b \in [0, 1]`.

    Args:
        a (float, optional): Amplitude parameter. Default: ``0.5``
        b (float, optional): Inner sine scaling parameter. Default: ``0.5``
        iterations (int, optional): Number of iterations for the chaotic map. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.CCAF()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.CCAF(a=0.8, b=0.7, iterations=3)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.5, b: float = 0.5, iterations: int = 1):
        super().__init__()
        assert 0 <= a <= 1, "Parameter 'a' must be in the range [0, 1]"
        assert 0 <= b <= 1, "Parameter 'b' must be in the range [0, 1]"
        self.a = a
        self.b = b
        self.iterations = iterations

    def forward(self, x: Tensor) -> Tensor:
        # Normalize input to [-1,1] range for sine stability
        z = torch.tanh(x)
        
        for _ in range(self.iterations):
            # Apply the cascade chaotic map
            z = self.a * torch.sin(torch.pi * self.b * torch.sin(torch.pi * z))
            
        return z