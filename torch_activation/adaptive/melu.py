import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

from torch_activation import register_activation

@register_activation
class MeLU(nn.Module):
    r"""
    Applies the Mexican ReLU (MeLU) function:

    :math:`\text{MeLU}(z_i) = \text{PReLU}(z_i) + \sum_{j=1}^{k-1} a_{i,j} \phi_{b_j c_j}(z_i)`

    where:
    :math:`\phi_{b_j c_j}(z_i) = \max(c_j - |z_i - b_j|, 0)`

    and :math:`a_{i,j}` are trainable parameters, :math:`b_j` and :math:`c_j` are fixed constants.

    Args:
        k (int, optional): Number of trainable parameters (k-1 for the sum and one for PReLU). Default: 4
        init_negative_slope (float, optional): Initial value for the PReLU negative slope. Default: 0.01
        init_a (float, optional): Initial value for the trainable parameters a. Default: 0.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MeLU(k=4)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, k: int = 4, init_negative_slope: float = 0.01, init_a: float = 0.0):
        super(MeLU, self).__init__()
        self.k = k
        
        # PReLU parameter
        self.prelu_weight = nn.Parameter(Tensor([init_negative_slope]))
        
        # Trainable parameters a_i,j
        self.a = nn.Parameter(torch.full((k-1,), init_a))
        
        # Fixed parameters b_j and c_j
        # Initialize them recursively as mentioned in the paper
        self.b = torch.zeros(k-1)
        self.c = torch.zeros(k-1)
        
        # Initialize b_j and c_j recursively
        # This is a simple initialization scheme; the paper may have a more specific one
        for j in range(k-1):
            self.b[j] = j * 2.0 / (k-1) - 1.0  # Spread between -1 and 1
            self.c[j] = 1.0 / (j+1)            # Decreasing values

    def forward(self, x) -> Tensor:
        # PReLU part
        prelu_out = F.prelu(x, self.prelu_weight)
        
        # Sum part
        sum_part = torch.zeros_like(x)
        for j in range(self.k-1):
            # Calculate phi_b_j,c_j(z_i)
            phi = torch.clamp(self.c[j] - torch.abs(x - self.b[j]), min=0.0)
            sum_part += self.a[j] * phi
        
        return prelu_out + sum_part


@register_activation
class MMeLU(nn.Module):
    r"""
    Applies the Modified Mexican ReLU (MMeLU) function:

    :math:`\text{MMeLU}(z_i) = a_i \cdot \max(b_i - |z_i - c_i|, 0) + (1 - a_i) \cdot \text{ReLU}(z_i)`

    where :math:`a_i \in [0, 1]`, :math:`b_i \in \mathbb{R}^+`, and :math:`c_i \in \mathbb{R}` are trainable parameters.

    Args:
        init_a (float, optional): Initial value for parameter a. Default: 0.5
        init_b (float, optional): Initial value for parameter b. Default: 1.0
        init_c (float, optional): Initial value for parameter c. Default: 0.0
        inplace (bool, optional): Can optionally do the operation in-place for ReLU. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MMeLU(init_a=0.5, init_b=1.0, init_c=0.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, init_a: float = 0.5, init_b: float = 1.0, init_c: float = 0.0, inplace: bool = False):
        super(MMeLU, self).__init__()
        self.a_raw = nn.Parameter(Tensor([init_a]))
        self.b_raw = nn.Parameter(Tensor([init_b]))
        self.c = nn.Parameter(Tensor([init_c]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        # Constrain a to [0, 1] using sigmoid
        a = torch.sigmoid(self.a_raw)
        
        # Constrain b to be positive using softplus
        b = F.softplus(self.b_raw)
        
        # First term: a * max(b - |z - c|, 0)
        first_term = a * torch.clamp(b - torch.abs(x - self.c), min=0.0)
        
        # Second term: (1 - a) * ReLU(z)
        second_term = (1 - a) * F.relu(x, inplace=self.inplace)
        
        return first_term + second_term


@register_activation
class GaLU(nn.Module):
    r"""
    Applies the Gaussian ReLU (GaLU) function:

    :math:`\text{GaLU}(z_i) = \text{PReLU}(z_i) + \sum_{j=1}^{k-1} a_{i,j} \phi_{b_j c_j}(z_i)`

    where:
    :math:`\phi_{b_j c_j}(z_i) = \max(c_j - |z_i - b_j|, 0) + \min(|z - b_j - 2c_j| - c_j, 0)`

    and :math:`a_{i,j}` are trainable parameters, :math:`b_j` and :math:`c_j` are fixed constants.

    Args:
        k (int, optional): Number of trainable parameters (k-1 for the sum and one for PReLU). Default: 4
        init_negative_slope (float, optional): Initial value for the PReLU negative slope. Default: 0.01
        init_a (float, optional): Initial value for the trainable parameters a. Default: 0.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = GaLU(k=4)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, k: int = 4, init_negative_slope: float = 0.01, init_a: float = 0.0):
        super(GaLU, self).__init__()
        self.k = k
        
        # PReLU parameter
        self.prelu_weight = nn.Parameter(Tensor([init_negative_slope]))
        
        # Trainable parameters a_i,j
        self.a = nn.Parameter(torch.full((k-1,), init_a))
        
        # Fixed parameters b_j and c_j
        # Initialize them similarly to MeLU
        self.b = torch.zeros(k-1)
        self.c = torch.zeros(k-1)
        
        # Initialize b_j and c_j
        for j in range(k-1):
            self.b[j] = j * 2.0 / (k-1) - 1.0  # Spread between -1 and 1
            self.c[j] = 1.0 / (j+1)            # Decreasing values

    def forward(self, x) -> Tensor:
        # PReLU part
        prelu_out = F.prelu(x, self.prelu_weight)
        
        # Sum part
        sum_part = torch.zeros_like(x)
        for j in range(self.k-1):
            # Calculate phi_b_j,c_j(z_i) for GaLU
            term1 = torch.clamp(self.c[j] - torch.abs(x - self.b[j]), min=0.0)
            term2 = torch.clamp(torch.abs(x - self.b[j] - 2*self.c[j]) - self.c[j], max=0.0)
            phi = term1 + term2
            sum_part += self.a[j] * phi
        
        return prelu_out + sum_part


@register_activation
class HardSwish(nn.Module):
    r"""
    Applies the Hard-Swish function:

    :math:`\text{HardSwish}(z_i) = 2z_i \cdot \max(0, \min(0.2b_i z_i + 0.5, 1))`

    where :math:`b_i` is a trainable parameter.

    Args:
        b_init (float, optional): Initial value for the trainable parameter b. Default: 1.0
        inplace (bool, optional): Can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardSwish(b_init=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, b_init: float = 1.0, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.b = nn.Parameter(Tensor([b_init]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        # Calculate hard sigmoid part: max(0, min(0.2*b*x + 0.5, 1))
        hard_sigmoid = torch.clamp(0.2 * self.b * x + 0.5, min=0.0, max=1.0)
        
        # Multiply by 2*x
        if self.inplace and x.is_floating_point():
            result = x.mul_(2).mul_(hard_sigmoid)
            return result
        else:
            return 2 * x * hard_sigmoid
