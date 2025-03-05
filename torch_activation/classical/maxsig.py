import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_activation import register_activation


@register_activation
class TanhLinearUnit(nn.Module):
    r"""
    Applies the Tanh Linear Unit activation function:

    :math:`\text{TanhLinearUnit}(z) = \begin{cases} 
    z, & z \geq 0 \\
    \frac{2}{1 + \exp(-z)} - 1, & z < 0 
    \end{cases} = \begin{cases} 
    z, & z \geq 0 \\
    \tanh\left(\frac{z}{2}\right), & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TanhLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(TanhLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = x.clone()
        result[neg_mask] = torch.tanh(x[neg_mask] / 2)
        
        return result


class DualELU(nn.Module):
    r"""
    Applies the Dual ELU activation function:

    :math:`\text{DualELU}(z, z') = \text{ELU}(z) - \text{ELU}(z')`

    Args:
        alpha (float, optional): The alpha value for the ELU formulation. Default: 1.0
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = DualELU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, alpha: float = 1.0, dim: int = -1):
        super(DualELU, self).__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, x) -> Tensor:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        z, z_prime = torch.split(x, split_size, dim=self.dim)
        
        return F.elu(z, alpha=self.alpha) - F.elu(z_prime, alpha=self.alpha)


@register_activation
class DifferenceELU(nn.Module):
    r"""
    Applies the Difference ELU activation function:

    :math:`\text{DifferenceELU}(z) = \begin{cases} 
    z, & z \geq 0 \\
    a(z\exp(z) - b\exp(bz)), & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Exponential scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = DifferenceELU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(DifferenceELU, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = x.clone()
        neg_x = x[neg_mask]
        result[neg_mask] = self.a * (neg_x * torch.exp(neg_x) - self.b * torch.exp(self.b * neg_x))
        
        return result


@register_activation
class PolynomialLinearUnit(nn.Module):
    r"""
    Applies the Polynomial Linear Unit activation function:

    :math:`\text{PolynomialLinearUnit}(z) = \begin{cases} 
    z, & z \geq 0 \\
    \frac{1}{1 - z} - 1, & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = PolynomialLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(PolynomialLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = x.clone()
        neg_x = x[neg_mask]
        
        # Ensure numerical stability by clamping values
        neg_x = torch.clamp(neg_x, min=-0.999)
        result[neg_mask] = 1 / (1 - neg_x) - 1
        
        return result


@register_activation
class InversePolynomialLinearUnit(nn.Module):
    r"""
    Applies the Inverse Polynomial Linear Unit activation function:

    :math:`\text{InversePolynomialLinearUnit}(z) = \begin{cases} 
    z, & z \geq 0 \\
    \frac{1}{1 + |z|^a}, & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Power parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = InversePolynomialLinearUnit(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(InversePolynomialLinearUnit, self).__init__()
        self.a = nn.Parameter(Tensor([a]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = x.clone()
        neg_x = x[neg_mask]
        result[neg_mask] = 1 / (1 + torch.abs(neg_x).pow(self.a))
        
        return result


@register_activation
class PowerLinearUnit(nn.Module):
    r"""
    Applies the Power Linear Unit activation function:

    :math:`\text{PowerLinearUnit}(z) = \begin{cases} 
    z, & z \geq 0 \\
    (1 - z)^{-a} - 1, & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Power parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = PowerLinearUnit(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(PowerLinearUnit, self).__init__()
        self.a = nn.Parameter(Tensor([a]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = x.clone()
        neg_x = x[neg_mask]
        
        # Ensure numerical stability by clamping values
        neg_x = torch.clamp(neg_x, min=-0.999)
        result[neg_mask] = torch.pow(1 - neg_x, -self.a) - 1
        
        return result


@register_activation
class PowerFunctionLinearUnit(nn.Module):
    r"""
    Applies the Power Function Linear Unit activation function:

    :math:`\text{PowerFunctionLinearUnit}(z) = z \cdot \frac{1}{2} \left( 1 + \frac{z}{\sqrt{1 + z^2}} \right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = PowerFunctionLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(PowerFunctionLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        return x * 0.5 * (1 + x / torch.sqrt(1 + x.pow(2)))


@register_activation
class FasterPowerFunctionLinearUnit(nn.Module):
    r"""
    Applies the Faster Power Function Linear Unit activation function:

    :math:`\text{FasterPowerFunctionLinearUnit}(z) = \begin{cases} 
    z, & z \geq 0 \\
    z + \frac{z^2}{\sqrt{1 + z^2}}, & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = FasterPowerFunctionLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(FasterPowerFunctionLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = x.clone()
        neg_x = x[neg_mask]
        result[neg_mask] = neg_x + (neg_x.pow(2) / torch.sqrt(1 + neg_x.pow(2)))
        
        return result


@register_activation
class ElasticAdaptivelyParametricCompoundedUnit(nn.Module):
    r"""
    Applies the Elastic Adaptively Parametric Compounded Unit activation function:

    :math:`\text{ElasticAdaptivelyParametricCompoundedUnit}(z_i) = \begin{cases} 
    b_i z_i, & z_i \geq 0 \\
    a_i z_i \cdot \tanh(\ln(1 + \exp(a_{i}z_{i}))), & z_i < 0 
    \end{cases}`

    Args:
        a (float or Tensor, optional): Negative slope parameter. Default: 1.0
        b (float or Tensor, optional): Positive slope parameter. Default: 1.0
        num_parameters (int, optional): Number of parameters if using per-channel parameterization. Default: 1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ElasticAdaptivelyParametricCompoundedUnit(a=0.5, b=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
        
        >>> # Per-channel parameterization
        >>> m = ElasticAdaptivelyParametricCompoundedUnit(num_parameters=3)
        >>> x = torch.randn(3, 5)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, num_parameters: int = 1):
        super(ElasticAdaptivelyParametricCompoundedUnit, self).__init__()
        self.num_parameters = num_parameters
        
        if num_parameters == 1:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = nn.Parameter(torch.full((num_parameters,), a))
            self.b = nn.Parameter(torch.full((num_parameters,), b))

    def forward(self, x) -> Tensor:
        if self.num_parameters == 1:
            pos_mask = x >= 0
            neg_mask = ~pos_mask
            
            result = torch.zeros_like(x)
            result[pos_mask] = self.b * x[pos_mask]
            
            neg_x = x[neg_mask]
            softplus = torch.log(1 + torch.exp(self.a * neg_x))
            result[neg_mask] = self.a * neg_x * torch.tanh(softplus)
            
            return result
        else:
            # Handle per-channel parameterization
            pos_mask = x >= 0
            neg_mask = ~pos_mask
            
            result = torch.zeros_like(x)
            
            for i in range(self.num_parameters):
                # Apply positive part
                channel_pos_mask = pos_mask.narrow(0, i, 1).squeeze(0)
                if channel_pos_mask.any():
                    result.narrow(0, i, 1)[channel_pos_mask] = self.b[i] * x.narrow(0, i, 1)[channel_pos_mask]
                
                # Apply negative part
                channel_neg_mask = neg_mask.narrow(0, i, 1).squeeze(0)
                if channel_neg_mask.any():
                    neg_x = x.narrow(0, i, 1)[channel_neg_mask]
                    softplus = torch.log(1 + torch.exp(self.a[i] * neg_x))
                    result.narrow(0, i, 1)[channel_neg_mask] = self.a[i] * neg_x * torch.tanh(softplus)
            
            return result


@register_activation
class LipschitzReLU(nn.Module):
    r"""
    Applies the Lipschitz ReLU activation function:

    :math:`\text{LipschitzReLU}(z) = p(z | z > 0) + n(z | z \leq 0)`

    where p and n are positive and negative functions with Lipschitz constant <= 1.

    Args:
        p_fn (callable, optional): Positive function. Default: identity
        n_fn (callable, optional): Negative function. Default: zero

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> # Default implementation (standard ReLU)
        >>> m = LipschitzReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
        
        >>> # Custom implementation with leaky behavior
        >>> m = LipschitzReLU(p_fn=lambda x: x, n_fn=lambda x: 0.1 * x)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, p_fn=None, n_fn=None):
        super(LipschitzReLU, self).__init__()
        self.p_fn = p_fn if p_fn is not None else lambda x: x
        self.n_fn = n_fn if n_fn is not None else lambda x: torch.zeros_like(x)

    def forward(self, x) -> Tensor:
        pos_mask = x > 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        result[pos_mask] = self.p_fn(x[pos_mask])
        result[neg_mask] = self.n_fn(x[neg_mask])
        
        return result


@register_activation
class ScaledExponentialLinearUnit(nn.Module):
    r"""
    Applies the Scaled Exponential Linear Unit activation function:

    :math:`\text{ScaledExponentialLinearUnit}(z) = \begin{cases} 
    az, & z \geq 0 \\
    ab(\exp(z) - 1), & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.67326
        b (float, optional): Alpha parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ScaledExponentialLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.67326, b: float = 1.0):
        super(ScaledExponentialLinearUnit, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x) -> Tensor:
        return F.selu(x)


@register_activation
class LeakyScaledExponentialLinearUnit(nn.Module):
    r"""
    Applies the Leaky Scaled Exponential Linear Unit activation function:

    :math:`\text{LeakyScaledExponentialLinearUnit}(z) = \begin{cases} 
    az, & z \geq 0 \\
    ab(\exp(z) - 1) + acz, & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Alpha parameter. Default: 1.0
        c (float, optional): Leaky slope. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LeakyScaledExponentialLinearUnit(a=1.5, b=1.0, c=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.1):
        super(LeakyScaledExponentialLinearUnit, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))
        self.c = nn.Parameter(Tensor([c]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        result[pos_mask] = self.a * x[pos_mask]
        
        neg_x = x[neg_mask]
        result[neg_mask] = self.a * self.b * (torch.exp(neg_x) - 1) + self.a * self.c * neg_x
        
        return result


@register_activation
class ScaledExponentiallyRegularizedLinearUnit(nn.Module):
    r"""
    Applies the Scaled Exponentially Regularized Linear Unit activation function:

    :math:`\text{ScaledExponentiallyRegularizedLinearUnit}(z) = \begin{cases} 
    az, & z \geq 0 \\
    abz\exp(z), & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Regularization parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ScaledExponentiallyRegularizedLinearUnit(a=1.5, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(ScaledExponentiallyRegularizedLinearUnit, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        result[pos_mask] = self.a * x[pos_mask]
        
        neg_x = x[neg_mask]
        result[neg_mask] = self.a * self.b * neg_x * torch.exp(neg_x)
        
        return result


@register_activation
class ScaledScaledExponentialLinearUnit(nn.Module):
    r"""
    Applies the Scaled Scaled Exponential Linear Unit activation function:

    :math:`\text{ScaledScaledExponentialLinearUnit}(z) = \begin{cases} 
    az, & z \geq 0 \\
    ab(\exp(cz) - 1), & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Alpha parameter. Default: 1.0
        c (float, optional): Exponential scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ScaledScaledExponentialLinearUnit(a=1.5, b=1.0, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        super(ScaledScaledExponentialLinearUnit, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))
        self.c = nn.Parameter(Tensor([c]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        result[pos_mask] = self.a * x[pos_mask]
        
        neg_x = x[neg_mask]
        result[neg_mask] = self.a * self.b * (torch.exp(self.c * neg_x) - 1)
        
        return result


@register_activation
class RSigELU(nn.Module):
    r"""
    Applies the RSigELU activation function:

    :math:`\text{RSigELU}(z) = \begin{cases} 
    z \cdot \frac{1}{1 + \exp(-z)} a + z, & 1 < z < \infty \\
    z, & 0 \geq z \geq 1 \\
    a(\exp(z) - 1), & -\infty < z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = RSigELU(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(RSigELU, self).__init__()
        self.a = nn.Parameter(Tensor([a]))

    def forward(self, x) -> Tensor:
        result = torch.zeros_like(x)
        
        # Case 1: 1 < z < infinity
        mask1 = x > 1
        result[mask1] = x[mask1] * torch.sigmoid(x[mask1]) * self.a + x[mask1]
        
        # Case 2: 0 <= z <= 1
        mask2 = (x >= 0) & (x <= 1)
        result[mask2] = x[mask2]
        
        # Case 3: -infinity < z < 0
        mask3 = x < 0
        result[mask3] = self.a * (torch.exp(x[mask3]) - 1)
        
        return result


@register_activation
class HardSReLUE(nn.Module):
    r"""
    Applies the Hard SReLUE activation function:

    :math:`\text{HardSReLUE}(z) = \begin{cases} 
    az \cdot \max\left(0, \min\left(1, \frac{z+1}{2} + z\right)\right), & z \geq 0 \\
    a(\exp(z) - 1), & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardSReLUE(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(HardSReLUE, self).__init__()
        self.a = nn.Parameter(Tensor([a]))

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        
        # Positive part
        pos_x = x[pos_mask]
        hard_sigmoid = torch.clamp((pos_x + 1) / 2 + pos_x, 0, 1)
        result[pos_mask] = self.a * pos_x * hard_sigmoid
        
        # Negative part
        neg_x = x[neg_mask]
        result[neg_mask] = self.a * (torch.exp(neg_x) - 1)
        
        return result


@register_activation
class ExponentialLinearSigmoidSquashing(nn.Module):
    r"""
    Applies the Exponential Linear Sigmoid Squashing activation function:

    :math:`\text{ExponentialLinearSigmoidSquashing}(z) = \begin{cases} 
    \frac{z}{1 + \exp(-z)}, & z \geq 0 \\
    \frac{\exp(z) - 1}{1 + \exp(-z)}, & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ExponentialLinearSigmoidSquashing()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ExponentialLinearSigmoidSquashing, self).__init__()

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        sigmoid = torch.sigmoid(x)
        
        # Positive part
        result[pos_mask] = x[pos_mask] * sigmoid[pos_mask]
        
        # Negative part
        neg_x = x[neg_mask]
        result[neg_mask] = (torch.exp(neg_x) - 1) * sigmoid[neg_mask]
        
        return result


@register_activation
class HardExponentialLinearSigmoidSquashing(nn.Module):
    r"""
    Applies the Hard Exponential Linear Sigmoid Squashing activation function:

    :math:`\text{HardExponentialLinearSigmoidSquashing}(z) = \begin{cases} 
    z \cdot \max\left(0, \min\left(\frac{z+1}{2}, 1\right)\right), & z \geq 0 \\
    (1 + \exp(-z)) \cdot \max\left(0, \min\left(\frac{z+1}{2}, 1\right)\right), & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardExponentialLinearSigmoidSquashing()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(HardExponentialLinearSigmoidSquashing, self).__init__()

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        hard_sigmoid = torch.clamp((x + 1) / 2, 0, 1)
        
        # Positive part
        result[pos_mask] = x[pos_mask] * hard_sigmoid[pos_mask]
        
        # Negative part
        neg_x = x[neg_mask]
        result[neg_mask] = (1 + torch.exp(-neg_x)) * hard_sigmoid[neg_mask]
        
        return result


@register_activation
class RSigELUD(nn.Module):
    r"""
    Applies the RSigELUD activation function:

    :math:`\text{RSigELUD}(z) = \begin{cases} 
    z \cdot \frac{1}{1 + \exp(-z)} a + z, & 1 < z < \infty \\
    z, & 0 \leq z \leq 1 \\
    b(\exp(z) - 1), & -\infty < z < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for z > 1. Default: 1.0
        b (float, optional): Scale parameter for z < 0. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = RSigELUD(a=1.5, b=0.8)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(RSigELUD, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def forward(self, x) -> Tensor:
        result = torch.zeros_like(x)
        
        # Case 1: 1 < z < infinity
        mask1 = x > 1
        result[mask1] = x[mask1] * torch.sigmoid(x[mask1]) * self.a + x[mask1]
        
        # Case 2: 0 <= z <= 1
        mask2 = (x >= 0) & (x <= 1)
        result[mask2] = x[mask2]
        
        # Case 3: -infinity < z < 0
        mask3 = x < 0
        result[mask3] = self.b * (torch.exp(x[mask3]) - 1)
        
        return result


@register_activation 
class LSReLU(nn.Module):
    r"""
    Applies the LSReLU activation function:

    :math:`\text{LSReLU}(z) = \begin{cases} 
    \frac{z}{1 + |z|}, & z \leq 0 \\
    z, & 0 \leq z \leq b \\
    \log(az + 1) + |\log(ab + 1) - b|, & z \geq b 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for z > b. Default: 1.0
        b (float, optional): Threshold parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LSReLU(a=0.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(LSReLU, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def forward(self, x) -> Tensor:
        result = torch.zeros_like(x)
        
        # Case 1: z <= 0
        mask1 = x <= 0
        neg_x = x[mask1]
        result[mask1] = neg_x / (1 + torch.abs(neg_x))
        
        # Case 2: 0 <= z <= b
        mask2 = (x > 0) & (x <= self.b)
        result[mask2] = x[mask2]
        
        # Case 3: z >= b
        mask3 = x > self.b
        log_term = torch.log(self.a * x[mask3] + 1)
        offset = torch.abs(torch.log(self.a * self.b + 1) - self.b)
        result[mask3] = log_term + offset
        
        return result