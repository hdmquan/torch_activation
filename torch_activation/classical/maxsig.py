import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class TanhLinearUnit(BaseActivation):
    r"""
    Applies the Tanh Linear Unit activation function:

    .. math::

        \text{TanhLinearUnit}(z) = \begin{cases}
        z, & z \geq 0 \\
        \tanh\left(\frac{z}{2}\right), & z < 0
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TanhLinearUnit.png

    Examples::

        >>> m = torch_activation.TanhLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = x.clone()
        result[neg_mask] = torch.tanh(x[neg_mask] / 2)

        return result


class DualELU(BaseActivation):
    r"""
    Applies the Dual ELU activation function:

    :math:`\text{DualELU}(z, z') = \text{ELU}(z) - \text{ELU}(z')`

    Args:
        alpha (float, optional): alpha value for the ELU formulation. Default: ``1.0``
        dim (int, optional): dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*, N, *)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, N/2, *)`, same shape as the input but halved along dim.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DualELU.png

    Examples::

        >>> m = torch_activation.DualELU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, alpha: float = 1.0, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.dim = dim

    def _forward(self, x) -> Tensor:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"

        split_size = dim_size // 2
        z, z_prime = torch.split(x, split_size, dim=self.dim)

        return F.elu(z, alpha=self.alpha) - F.elu(z_prime, alpha=self.alpha)


@register_activation
class DifferenceELU(BaseActivation):
    r"""
    Applies the Difference ELU activation function:

    .. math::

        \text{DifferenceELU}(z) = \begin{cases}
        z, & z \geq 0 \\
        a(z\exp(z) - b\exp(bz)), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0``
        b (float, optional): exponential scale parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DifferenceELU.png

    Examples::

        >>> m = torch_activation.DifferenceELU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def _forward(self, x) -> Tensor:
        a, b = self.a.to(x.dtype), self.b.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = x.clone()
        neg_x = x[neg_mask]
        result[neg_mask] = a * (neg_x * torch.exp(neg_x) - b * torch.exp(b * neg_x))

        return result


@register_activation
class PolynomialLinearUnit(BaseActivation):
    r"""
    Applies the Polynomial Linear Unit activation function:

    .. math::

        \text{PolynomialLinearUnit}(z) = \begin{cases}
        z, & z \geq 0 \\
        \frac{1}{1 - z} - 1, & z < 0
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PolynomialLinearUnit.png

    Examples::

        >>> m = torch_activation.PolynomialLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = x.clone()
        neg_x = x[neg_mask]

        # Ensure numerical stability by clamping values
        neg_x = torch.clamp(neg_x, min=-0.999)
        result[neg_mask] = 1 / (1 - neg_x) - 1

        return result


@register_activation
class InversePolynomialLinearUnit(BaseActivation):
    r"""
    Applies the Inverse Polynomial Linear Unit activation function:

    .. math::

        \text{InversePolynomialLinearUnit}(z) = \begin{cases}
        z, & z \geq 0 \\
        \frac{1}{1 + |z|^a}, & z < 0
        \end{cases}

    Args:
        a (float, optional): power parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/InversePolynomialLinearUnit.png

    Examples::

        >>> m = torch_activation.InversePolynomialLinearUnit(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))

    def _forward(self, x) -> Tensor:
        a = self.a.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = x.clone()
        neg_x = x[neg_mask]
        result[neg_mask] = 1 / (1 + torch.abs(neg_x).pow(a))

        return result


@register_activation
class PowerLinearUnit(BaseActivation):
    r"""
    Applies the Power Linear Unit activation function:

    .. math::

        \text{PowerLinearUnit}(z) = \begin{cases}
        z, & z \geq 0 \\
        (1 - z)^{-a} - 1, & z < 0
        \end{cases}

    Args:
        a (float, optional): power parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PowerLinearUnit.png

    Examples::

        >>> m = torch_activation.PowerLinearUnit(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))

    def _forward(self, x) -> Tensor:
        a = self.a.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = x.clone()
        neg_x = x[neg_mask]

        neg_x = torch.clamp(neg_x, min=-0.999)
        result[neg_mask] = torch.pow(1 - neg_x, -a) - 1

        return result


@register_activation
class PowerFunctionLinearUnit(BaseActivation):
    r"""
    Applies the Power Function Linear Unit activation function:

    :math:`\text{PowerFunctionLinearUnit}(z) = z \cdot \frac{1}{2} \left( 1 + \frac{z}{\sqrt{1 + z^2}} \right)`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PowerFunctionLinearUnit.png

    Examples::

        >>> m = torch_activation.PowerFunctionLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * 0.5 * (1 + x / torch.sqrt(1 + x.pow(2)))


@register_activation
class FasterPowerFunctionLinearUnit(BaseActivation):
    r"""
    Applies the Faster Power Function Linear Unit activation function:

    .. math::

        \text{FasterPowerFunctionLinearUnit}(z) = \begin{cases}
        z, & z \geq 0 \\
        z + \frac{z^2}{\sqrt{1 + z^2}}, & z < 0
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/FasterPowerFunctionLinearUnit.png

    Examples::

        >>> m = torch_activation.FasterPowerFunctionLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = x.clone()
        neg_x = x[neg_mask]
        result[neg_mask] = neg_x + (neg_x.pow(2) / torch.sqrt(1 + neg_x.pow(2)))

        return result


@register_activation
class ElasticAdaptivelyParametricCompoundedUnit(BaseActivation):
    r"""
    Applies the Elastic Adaptively Parametric Compounded Unit activation function:

    .. math::

        \text{ElasticAdaptivelyParametricCompoundedUnit}(z_i) = \begin{cases}
        b_i z_i, & z_i \geq 0 \\
        a_i z_i \cdot \tanh(\ln(1 + \exp(a_{i}z_{i}))), & z_i < 0
        \end{cases}

    Args:
        a (float, optional): negative slope parameter. Default: ``1.0``
        b (float, optional): positive slope parameter. Default: ``1.0``
        num_parameters (int, optional): number of per-channel parameters. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ElasticAdaptivelyParametricCompoundedUnit.png

    Examples::

        >>> m = torch_activation.ElasticAdaptivelyParametricCompoundedUnit(a=0.5, b=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, num_parameters: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_parameters = num_parameters

        if num_parameters == 1:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = nn.Parameter(torch.full((num_parameters,), a))
            self.b = nn.Parameter(torch.full((num_parameters,), b))

    def _forward(self, x) -> Tensor:
        if self.num_parameters == 1:
            a, b = self.a.to(x.dtype), self.b.to(x.dtype)
            pos_mask = x >= 0
            neg_mask = ~pos_mask

            result = torch.zeros_like(x)
            result[pos_mask] = b * x[pos_mask]

            neg_x = x[neg_mask]
            softplus = torch.log(1 + torch.exp(a * neg_x))
            result[neg_mask] = a * neg_x * torch.tanh(softplus)

            return result
        else:
            a = self.a.to(x.dtype)
            b = self.b.to(x.dtype)
            pos_mask = x >= 0
            neg_mask = ~pos_mask

            result = torch.zeros_like(x)

            for i in range(self.num_parameters):
                channel_pos_mask = pos_mask.narrow(0, i, 1).squeeze(0)
                if channel_pos_mask.any():
                    result.narrow(0, i, 1)[channel_pos_mask] = (
                        b[i] * x.narrow(0, i, 1)[channel_pos_mask]
                    )

                channel_neg_mask = neg_mask.narrow(0, i, 1).squeeze(0)
                if channel_neg_mask.any():
                    neg_x = x.narrow(0, i, 1)[channel_neg_mask]
                    softplus = torch.log(1 + torch.exp(a[i] * neg_x))
                    result.narrow(0, i, 1)[channel_neg_mask] = a[i] * neg_x * torch.tanh(softplus)

            return result


@register_activation
class LipschitzReLU(BaseActivation):
    r"""
    Applies the Lipschitz ReLU activation function:

    :math:`\text{LipschitzReLU}(z) = p(z)\,[z > 0] + n(z)\,[z \leq 0]`

    where :math:`p` and :math:`n` are functions with Lipschitz constant at most 1.

    Args:
        p_fn (callable, optional): positive-region function. Default: ``identity``
        n_fn (callable, optional): negative-region function. Default: ``zero``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LipschitzReLU.png

    Examples::

        >>> m = torch_activation.LipschitzReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, p_fn=None, n_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.p_fn = p_fn if p_fn is not None else lambda x: x
        self.n_fn = n_fn if n_fn is not None else lambda x: torch.zeros_like(x)

    def _forward(self, x) -> Tensor:
        pos_mask = x > 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)
        result[pos_mask] = self.p_fn(x[pos_mask])
        result[neg_mask] = self.n_fn(x[neg_mask])

        return result


@register_activation
class ScaledExponentialLinearUnit(BaseActivation):
    r"""
    Applies the Scaled Exponential Linear Unit activation function:

    .. math::

        \text{ScaledExponentialLinearUnit}(z) = \begin{cases}
        az, & z \geq 0 \\
        ab(\exp(z) - 1), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0507009873554804934193``
        b (float, optional): alpha parameter. Default: ``1.6732631921033945073073``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledExponentialLinearUnit.png

    Examples::

        >>> m = torch_activation.ScaledExponentialLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(
        self, a: float = 1.0507009873554804934193, b: float = 1.6732631921033945073073, **kwargs
    ):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        result = torch.zeros_like(x)
        result[pos_mask] = self.a * x[pos_mask]
        result[neg_mask] = self.a * self.b * (torch.exp(x[neg_mask]) - 1)
        return result


@register_activation
class LeakyScaledExponentialLinearUnit(BaseActivation):
    r"""
    Applies the Leaky Scaled Exponential Linear Unit activation function:

    .. math::

        \text{LeakyScaledExponentialLinearUnit}(z) = \begin{cases}
        az, & z \geq 0 \\
        ab(\exp(z) - 1) + acz, & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0``
        b (float, optional): alpha parameter. Default: ``1.0``
        c (float, optional): leaky slope. Default: ``0.1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LeakyScaledExponentialLinearUnit.png

    Examples::

        >>> m = torch_activation.LeakyScaledExponentialLinearUnit(a=1.5, b=1.0, c=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))
        self.c = nn.Parameter(Tensor([c]))

    def _forward(self, x) -> Tensor:
        a, b, c = self.a.to(x.dtype), self.b.to(x.dtype), self.c.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)
        result[pos_mask] = a * x[pos_mask]

        neg_x = x[neg_mask]
        result[neg_mask] = a * b * (torch.exp(neg_x) - 1) + a * c * neg_x

        return result


@register_activation
class ScaledExponentiallyRegularizedLinearUnit(BaseActivation):
    r"""
    Applies the Scaled Exponentially Regularized Linear Unit activation function:

    .. math::

        \text{ScaledExponentiallyRegularizedLinearUnit}(z) = \begin{cases}
        az, & z \geq 0 \\
        abz\exp(z), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0``
        b (float, optional): regularization parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledExponentiallyRegularizedLinearUnit.png

    Examples::

        >>> m = torch_activation.ScaledExponentiallyRegularizedLinearUnit(a=1.5, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def _forward(self, x) -> Tensor:
        a, b = self.a.to(x.dtype), self.b.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)
        result[pos_mask] = a * x[pos_mask]

        neg_x = x[neg_mask]
        result[neg_mask] = a * b * neg_x * torch.exp(neg_x)

        return result


@register_activation
class ScaledScaledExponentialLinearUnit(BaseActivation):
    r"""
    Applies the Scaled Scaled Exponential Linear Unit activation function:

    .. math::

        \text{ScaledScaledExponentialLinearUnit}(z) = \begin{cases}
        az, & z \geq 0 \\
        ab(\exp(cz) - 1), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0``
        b (float, optional): alpha parameter. Default: ``1.0``
        c (float, optional): exponential scale parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledScaledExponentialLinearUnit.png

    Examples::

        >>> m = torch_activation.ScaledScaledExponentialLinearUnit(a=1.5, b=1.0, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))
        self.c = nn.Parameter(Tensor([c]))

    def _forward(self, x) -> Tensor:
        a, b, c = self.a.to(x.dtype), self.b.to(x.dtype), self.c.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)
        result[pos_mask] = a * x[pos_mask]

        neg_x = x[neg_mask]
        result[neg_mask] = a * b * (torch.exp(c * neg_x) - 1)

        return result


@register_activation
class RSigELU(BaseActivation):
    r"""
    Applies the RSigELU activation function:

    .. math::

        \text{RSigELU}(z) = \begin{cases}
        z \cdot \sigma(z) \cdot a + z, & z > 1 \\
        z, & 0 \leq z \leq 1 \\
        a(\exp(z) - 1), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RSigELU.png

    Examples::

        >>> m = torch_activation.RSigELU(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))

    def _forward(self, x) -> Tensor:
        a = self.a.to(x.dtype)
        result = torch.zeros_like(x)

        mask1 = x > 1
        result[mask1] = x[mask1] * torch.sigmoid(x[mask1]) * a + x[mask1]

        mask2 = (x >= 0) & (x <= 1)
        result[mask2] = x[mask2]

        mask3 = x < 0
        result[mask3] = a * (torch.exp(x[mask3]) - 1)

        return result


@register_activation
class HardSReLUE(BaseActivation):
    r"""
    Applies the Hard SReLUE activation function:

    .. math::

        \text{HardSReLUE}(z) = \begin{cases}
        az \cdot \max\!\left(0, \min\!\left(1, \tfrac{z+1}{2}\right)\right) + z, & z \geq 0 \\
        a(\exp(z) - 1), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardSReLUE.png

    Examples::

        >>> m = torch_activation.HardSReLUE(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))

    def _forward(self, x) -> Tensor:
        a = self.a.to(x.dtype)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)

        pos_x = x[pos_mask]
        hard_sigmoid = torch.clamp((pos_x + 1) / 2, 0, 1)
        result[pos_mask] = a * pos_x * hard_sigmoid + pos_x

        neg_x = x[neg_mask]
        result[neg_mask] = a * (torch.exp(neg_x) - 1)

        return result


@register_activation
class ExponentialLinearSigmoidSquashing(BaseActivation):
    r"""
    Applies the Exponential Linear Sigmoid Squashing activation function:

    .. math::

        \text{ExponentialLinearSigmoidSquashing}(z) = \begin{cases}
        \frac{z}{1 + \exp(-z)}, & z \geq 0 \\
        \frac{\exp(z) - 1}{1 + \exp(-z)}, & z < 0
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ExponentialLinearSigmoidSquashing.png

    Examples::

        >>> m = torch_activation.ExponentialLinearSigmoidSquashing()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
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
class HardExponentialLinearSigmoidSquashing(BaseActivation):
    r"""
    Applies the Hard Exponential Linear Sigmoid Squashing activation function:

    .. math::

        \text{HardExponentialLinearSigmoidSquashing}(z) = \begin{cases}
        z \cdot \max\!\left(0, \min\!\left(\tfrac{z+1}{2}, 1\right)\right), & z \geq 0 \\
        (\exp(z) - 1) \cdot \max\!\left(0, \min\!\left(\tfrac{z+1}{2}, 1\right)\right), & z < 0
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardExponentialLinearSigmoidSquashing.png

    Examples::

        >>> m = torch_activation.HardExponentialLinearSigmoidSquashing()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        hard_sigmoid = torch.clamp((x + 1) / 2, 0, 1)
        neg_term = 1 + torch.exp(-x)
        raw = torch.where(pos_mask, x * hard_sigmoid, neg_term * hard_sigmoid)
        return torch.where(hard_sigmoid == 0, torch.zeros_like(x), raw)


@register_activation
class RSigELUD(BaseActivation):
    r"""
    Applies the RSigELUD activation function:

    .. math::

        \text{RSigELUD}(z) = \begin{cases}
        z \cdot \sigma(z) \cdot a + z, & z > 1 \\
        z, & 0 \leq z \leq 1 \\
        b(\exp(z) - 1), & z < 0
        \end{cases}

    Args:
        a (float, optional): scale parameter for :math:`z > 1`. Default: ``1.0``
        b (float, optional): scale parameter for :math:`z < 0`. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RSigELUD.png

    Examples::

        >>> m = torch_activation.RSigELUD(a=1.5, b=0.8)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def _forward(self, x) -> Tensor:
        a, b = self.a.to(x.dtype), self.b.to(x.dtype)
        result = torch.zeros_like(x)

        mask1 = x > 1
        result[mask1] = x[mask1] * torch.sigmoid(x[mask1]) * a + x[mask1]

        mask2 = (x >= 0) & (x <= 1)
        result[mask2] = x[mask2]

        mask3 = x < 0
        result[mask3] = b * (torch.exp(x[mask3]) - 1)

        return result


@register_activation
class LSReLU(BaseActivation):
    r"""
    Applies the LSReLU activation function:

    .. math::

        \text{LSReLU}(z) = \begin{cases}
        \frac{z}{1 + |z|}, & z \leq 0 \\
        z, & 0 \leq z \leq b \\
        \log(az + 1) + |\log(ab + 1) - b|, & z > b
        \end{cases}

    Args:
        a (float, optional): log-region scale parameter. Default: ``1.0``
        b (float, optional): threshold parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LSReLU.png

    Examples::

        >>> m = torch_activation.LSReLU(a=0.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]).squeeze())
        self.b = nn.Parameter(Tensor([b]).squeeze())

    def _forward(self, x) -> Tensor:
        offset = torch.abs(torch.log(self.a * self.b + 1) - self.b)
        neg_out = x / (1 + torch.abs(x))
        lin_out = x
        log_out = torch.log(self.a * x + 1) + offset
        return torch.where(x <= 0, neg_out, torch.where(x <= self.b, lin_out, log_out))


@register_activation
class Maxsig(BaseActivation):
    r"""
    Applies the Maxsig activation function:

    :math:`\text{Maxsig}(z) = \max(z, \sigma(z))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Maxsig.png

    Examples::

        >>> m = torch_activation.Maxsig()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.max(x, torch.sigmoid(x))
