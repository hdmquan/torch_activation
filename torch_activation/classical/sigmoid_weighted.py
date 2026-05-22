import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation
from torch_activation.utils import sech


# TODO: There are mentioned of WiG - a gated unit. Investigate it later..
@register_activation
class SiLU(BaseActivation):
    r"""
    Applies the Sigmoid Linear Unit activation function:

    :math:`\text{SiLU}(x) = x \cdot \sigma(x)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SiLU.png

    Examples::

        >>> m = torch_activation.SiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SiLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


@register_activation
class SinLU(BaseActivation):
    r"""
    Applies the Sinu-sigmoidal Linear Unit activation function:

    :math:`\text{SinLU}(x) = (x + a \cdot \sin (b \cdot x)) \sigma (x)`

     See: https://doi.org/10.3390/math10030337

    Args:
        a (float, optional): Initial value for sine function magnitude. Default: ``1.0``
        b (float, optional): Initial value for sine function period. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SinLU.png

    Examples::

        >>> m = torch_activation.SinLU(a=5.0, b=6.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SinLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.tensor(float(a)))
        self.beta = nn.Parameter(torch.tensor(float(b)))

    def _forward(self, x: Tensor) -> Tensor:
        result = x + self.alpha * torch.sin(self.beta * x)
        result *= torch.sigmoid(x)
        return result

    def _forward_inplace(self, x: Tensor) -> Tensor:
        s_x = torch.sigmoid(x)
        x.add_(self.alpha * torch.sin(self.beta * x))
        x.mul_(s_x)
        return x


@register_activation
class GELU(BaseActivation):
    r"""
    Applies the Gaussian Error Linear Unit activation function:

    :math:`\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2} \left( 1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right) \right)` # noqa: E501

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/GELU.png

    Examples::

        >>> m = torch_activation.GELU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GELU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


@register_activation
class SGELU(BaseActivation):
    r"""
    Applies the Symmetrical Gaussian Error Linear Unit activation function:

    :math:`\text{SGELU}(z) = a \cdot z \cdot \text{erf}\left(\frac{z}{\sqrt{2}}\right)`

    Args:
        a (float, optional): Scale parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SGELU.png

    Examples::

        >>> m = torch_activation.SGELU(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SGELU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(float(a)))

    def _forward(self, x: Tensor) -> Tensor:
        return self.a * x * torch.erf(x / math.sqrt(2))


@register_activation
class CaLU(BaseActivation):
    r"""
    Applies the Cauchy Linear Unit activation function:

    :math:`\text{CaLU}(z) = z \cdot \Phi_{\text{Cauchy}}(z) = z \cdot \left( \frac{\arctan(z)}{\pi} + \frac{1}{2} \right)` # noqa: E501

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/CaLU.png

    Examples::

        >>> m = torch_activation.CaLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.CaLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * (torch.arctan(x) / math.pi + 0.5)


@register_activation
class LaLU(BaseActivation):
    r"""
    Applies the Laplace Linear Unit activation function:

    .. math::

        \text{LaLU}(z) = z \cdot \Phi_{\text{Laplace}}(z) = z \cdot \begin{cases}
        1 - \frac{1}{2} \exp(-z), & z \geq 0 \\
        \frac{1}{2} \exp(z), & z < 0
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/LaLU.png

    Examples::

        >>> m = torch_activation.LaLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LaLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)
        result[pos_mask] = x[pos_mask] * (1 - 0.5 * torch.exp(-x[pos_mask]))
        result[neg_mask] = x[neg_mask] * (0.5 * torch.exp(x[neg_mask]))

        return result


# TODO: The paper mis-typed it as LaLU. Contact the author about it.
@register_activation
class CoLU(BaseActivation):
    r"""
    Applies the Collapsing Linear Unit activation function:

    :math:`\text{CoLU}(z) = z \cdot \frac{1}{1 - z \exp(-(z + \exp(z)))}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/CoLU.png

    Examples::

        >>> m = torch_activation.CoLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.CoLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False, **kwargs):
        super().__init__(inplace=inplace, **kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        denominator = 1 - x * torch.exp(-(x + torch.exp(x)))
        return x.div_(denominator) if self.inplace else x / denominator


@register_activation
class TSSwish(BaseActivation):
    r"""
    Applies the Triple State Swish activation function:

    :math:`\text{TSS}(z) = z \cdot \frac{1}{1 + \exp(-z)} \left( \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z+a)} + \frac{1}{1 + \exp(-z+b)} \right)` # noqa: E501

    Args:
        a (float, optional): First shift parameter. Default: ``1.0``
        b (float, optional): Second shift parameter. Default: ``2.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/TSSwish.png

    Examples::

        >>> m = torch_activation.TSSwish(a=1.5, b=2.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TSSwish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def _forward(self, x: Tensor) -> Tensor:
        # TODO: Memory
        sigmoid_x = torch.sigmoid(x)
        triple_term = sigmoid_x + torch.sigmoid(x - self.a) + torch.sigmoid(x - self.b)
        return x * sigmoid_x * triple_term


@register_activation
class GSwish(BaseActivation):
    r"""
    Applies the Generalized Swish activation function:

    :math:`\text{GSwish}(z) = z \cdot \sigma(\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/GSwish.png

    Examples::

        >>> m = torch_activation.GSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GSwish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(torch.exp(-x))


@register_activation
class ESwish(BaseActivation):
    r"""
    Applies the Exponential Swish activation function:

    :math:`\text{ESwish}(z) = \exp(-z) \cdot \sigma(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/ESwish.png

    Examples::

        >>> m = torch_activation.ESwish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ESwish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.exp((-x).clamp(max=88.0)) * torch.sigmoid(x)


@register_activation
class dSigmoid(BaseActivation):
    r"""
    Applies the Derivative of Sigmoid Function activation:

    :math:`\text{dSigmoid}(z) = \exp(-z) \cdot (\sigma(z))^2`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/dSigmoid.png

    Examples::

        >>> m = torch_activation.dSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.dSigmoid()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return torch.exp((-x).clamp(max=88.0)) * sigmoid_x * sigmoid_x


@register_activation
class Gish(BaseActivation):
    r"""
    Applies the Gish activation function:

    :math:`\text{Gish}(z) = z \cdot \ln(2 - \exp(-\exp(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Gish.png

    Examples::

        >>> m = torch_activation.Gish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Gish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.log(2 - torch.exp(-torch.exp(x)))


@register_activation
class Logish(BaseActivation):
    r"""
    Applies the Logish activation function:

    :math:`\text{Logish}(z) = z \cdot \ln(1 + \sigma(z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Logish.png

    Examples::

        >>> m = torch_activation.Logish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Logish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.log(1 + torch.sigmoid(x))


@register_activation
class LogLogish(BaseActivation):
    r"""
    Applies the LogLogish activation function:

    :math:`\text{LogLogish}(z) = z \cdot (1 - \exp(-\exp(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/LogLogish.png

    Examples::

        >>> m = torch_activation.LogLogish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LogLogish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * (1 - torch.exp(-torch.exp(x)))


@register_activation
class ExpExpish(BaseActivation):
    r"""
    Applies the ExpExpish activation function:

    :math:`\text{ExpExpish}(z) = z \cdot \exp(-\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/ExpExpish.png

    Examples::

        >>> m = torch_activation.ExpExpish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ExpExpish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.exp(-torch.exp(-x))


@register_activation
class SelfArctan(BaseActivation):
    r"""
    Applies the SelfArctan activation function:

    :math:`\text{SelfArctan}(z) = z \cdot \arctan(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SelfArctan.png

    Examples::

        >>> m = torch_activation.SelfArctan()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SelfArctan()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.arctan(x)


@register_activation
class pLogish(BaseActivation):
    r"""
    Applies the Parametric Logish activation function:

    :math:`\text{pLogish}(z_i) = a \cdot z_i \cdot \ln(1 + \sigma(b \cdot z_i))`

    Args:
        a (float, optional): Scale parameter. Default: ``1.0``
        b (float, optional): Sigmoid scale parameter. Default: ``10.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/pLogish.png

    Examples::

        >>> m = torch_activation.pLogish(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.pLogish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def _forward(self, x: Tensor) -> Tensor:
        return self.a * x * torch.log(1 + torch.sigmoid(self.b * x))


@register_activation
class Phish(BaseActivation):
    r"""
    Applies the Phish activation function:

    :math:`\text{Phish}(z) = z \cdot \tanh(\text{GELU}(z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Phish.png

    Examples::

        >>> m = torch_activation.Phish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Phish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.gelu(x))


@register_activation
class Suish(BaseActivation):
    r"""
    Applies the Suish activation function:

    :math:`\text{Suish}(z) = \max(z, z \cdot \exp(-|z|))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Suish.png

    Examples::

        >>> m = torch_activation.Suish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Suish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.maximum(x, x * torch.exp(-torch.abs(x)))


@register_activation
class TSReLU(BaseActivation):
    r"""
    Applies the Tangent Sigmoid ReLU activation function:

    :math:`\text{TSReLU}(z) = z \cdot \tanh(\sigma(z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/TSReLU.png

    Examples::

        >>> m = torch_activation.TSReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TSReLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.sigmoid(x))


@register_activation
class TBSReLU(BaseActivation):
    r"""
    Applies the Tangent Bipolar Sigmoid ReLU activation function:

    :math:`\text{TBSReLU}(z) = z \cdot \tanh\left(\frac{1 - \exp(-z)}{1 + \exp(-z)}\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/TBSReLU.png

    Examples::

        >>> m = torch_activation.TBSReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TBSReLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        exp_neg_x = torch.exp((-x).clamp(max=88.0))
        bipolar_sigmoid = (1 - exp_neg_x) / (1 + exp_neg_x)
        return x * torch.tanh(bipolar_sigmoid)


@register_activation
class LogSigmoid(BaseActivation):
    r"""
    Applies the LogSigmoid activation function:

    :math:`\text{LogSigmoid}(z) = \ln(\sigma(z)) = \ln\left(\frac{1}{1 + \exp(-z)}\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/LogSigmoid.png

    Examples::

        >>> m = torch_activation.LogSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LogSigmoid()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return F.logsigmoid(x)


@register_activation
class dSiLU(BaseActivation):
    r"""
    Applies the Derivative of SiLU activation function:

    :math:`\text{dSiLU}(z) = \sigma(z) \cdot (1 + z \cdot (1 - \sigma(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/dSiLU.png

    Examples::

        >>> m = torch_activation.dSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.dSiLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return sigmoid_x * (1 + x * (1 - sigmoid_x))


@register_activation
class DoubleSiLU(BaseActivation):
    r"""
    Applies the Double SiLU activation function:

    :math:`\text{DoubleSiLU}(z) = z \cdot \frac{1}{1 + \exp\left(-z \cdot \frac{1}{1 + \exp(-z)}\right)}` # noqa: E501

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/DoubleSiLU.png

    Examples::

        >>> m = torch_activation.DoubleSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DoubleSiLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(F.silu(x))


@register_activation
class MSiLU(BaseActivation):
    r"""
    Applies the Modified SiLU activation function:

    :math:`\text{MSiLU}(z) = z \cdot \sigma(z) + \exp\left(\frac{-z^2 - 1}{4}\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/MSiLU.png

    Examples::

        >>> m = torch_activation.MSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MSiLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x) + torch.exp((-x.pow(2) - 1) / 4)


@register_activation
class TSiLU(BaseActivation):
    r"""
    Applies the Hyperbolic Tangent Sigmoid-Weighted Linear Unit activation function:

    :math:`\text{TSiLU}(z) = \tanh\!\left(z \cdot \sigma(z)\right) = \frac{\exp\left(z \cdot \sigma(z)\right) - \exp\left(-z \cdot \sigma(z)\right)}{\exp\left(z \cdot \sigma(z)\right) + \exp\left(-z \cdot \sigma(z)\right)}` # noqa: E501

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/TSiLU.png

    Examples::

        >>> m = torch_activation.TSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TSiLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        # The paper wrote this in tanh exp form
        silu_x = x * torch.sigmoid(x)
        return torch.tanh(silu_x)


@register_activation
class ASiLU(BaseActivation):
    r"""
    Applies the Arctan SiLU activation function:

    :math:`\text{ASiLU}(z) = \arctan\left(z \cdot \frac{1}{1 + \exp(-z)}\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/ASiLU.png

    Examples::

        >>> m = torch_activation.ASiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ASiLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.arctan(x * torch.sigmoid(x))


# FIXME: Notation in paper is not clear, requires verification
# Should be done in a few days. Contact author when done.
@register_activation
class SwAT(BaseActivation):
    r"""
    Applies the SwAT activation function:

    :math:`\text{SwAT}(z) = z \cdot \frac{1}{1 + \exp(-\arctan(z))}`

     See: https://drive.google.com/file/d/10g-lrsc4WhxU90zQLaBY9BuYconaj-vD/view?usp=sharing

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SwAT.png

    Examples::

        >>> m = torch_activation.SwAT()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SwAT()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(torch.arctan(x))


@register_activation
class ReHSec(BaseActivation):
    r"""
    Applies the Rectified Hyperbolic Secant activation function:

    :math:`\text{ReHSec}(z) = z \cdot \text{sech}(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/ReHSec.png

    Examples::

        >>> m = torch_activation.ReHSec()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReHSec()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * sech(x)


@register_activation
class LiSHT(BaseActivation):
    r"""
    Applies the Linearly Scaled Hyperbolic Tangent activation function:

    :math:`\text{LiSHT}(z) = z \cdot \tanh(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/LiSHT.png

    Examples::

        >>> m = torch_activation.LiSHT()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LiSHT()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(x)


@register_activation
class Mish(BaseActivation):
    r"""
    Applies the Mish activation function:

    :math:`\text{Mish}(z) = z \cdot \tanh(\text{softplus}(z)) = z \cdot \tanh(\ln(1 + \exp(z)))`

     See: https://doi.org/10.48550/arXiv.1908.08681

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Mish.png

    Examples::

        >>> m = torch_activation.Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Mish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


@register_activation
class Smish(BaseActivation):
    r"""
    Applies the Smish activation function:

    :math:`\text{Smish}(z) = a \cdot z \cdot \tanh(\ln(1 + \sigma(b \cdot z)))`

    Args:
        a (float, optional): Scale parameter. Default: ``1.0``
        b (float, optional): Sigmoid scale parameter. Default: ``1.0``
        learnable (bool, optional): If True, the parameters are learnable. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Smish.png

    Examples::

        >>> m = torch_activation.Smish(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Smish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, learnable: bool = False, **kwargs):
        super().__init__(**kwargs)
        if learnable:
            self.a = nn.Parameter(torch.tensor(float(a)))
            self.b = nn.Parameter(torch.tensor(float(b)))
        else:
            self.register_buffer("a", torch.tensor(float(a)))
            self.register_buffer("b", torch.tensor(float(b)))

    def _forward(self, x: Tensor) -> Tensor:
        return self.a * x * torch.tanh(torch.log(1 + torch.sigmoid(self.b * x)))


@register_activation
class TanhExp(BaseActivation):
    r"""
    Applies the TanhExp activation function:

    :math:`\text{TanhExp}(z) = z \cdot \tanh(\exp(z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/TanhExp.png

    Examples::

        >>> m = torch_activation.TanhExp()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TanhExp()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.exp(x))


@register_activation
class Serf(BaseActivation):
    r"""
    Applies the SERF activation function:

    :math:`\text{Serf}(z) = z \cdot \text{erf}(\ln(1 + \exp(z)))`

     See: https://doi.org/10.48550/arXiv.2108.09598

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/Serf.png

    Examples::

        >>> m = torch_activation.Serf()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Serf()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.erf(F.softplus(x))


# NOTE: This can be a whole family my itself in the z * g(h(z)) form.
# In fact the functions should be customizable, but we just use the simplified version.
@register_activation
class EANAF(BaseActivation):
    r"""
    Applies the Efficient Asymmetric Nonlinear Activation Function:

    :math:`\text{EANAF}(z) = z \cdot \frac{\exp(z)}{\exp(z) + 2}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/EANAF.png

    Examples::

        >>> m = torch_activation.EANAF()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.EANAF()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        exp_x = torch.exp(x.clamp(max=88.0))
        return x * (exp_x / (exp_x + 2))


@register_activation
class SinSig(BaseActivation):
    r"""
    Applies the SinSig activation function:

    :math:`\text{SinSig}(z) = z \cdot \sin\left(\frac{\pi}{2} \sigma(z)\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SinSig.png

    Examples::

        >>> m = torch_activation.SinSig()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SinSig()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.sin((math.pi / 2) * torch.sigmoid(x))


@register_activation
class SiELU(BaseActivation):
    r"""
    Applies the Gaussian Error Linear Unit with Sigmoid Activation Functions:

    :math:`\text{SiELU}(z) = z \cdot \sigma\!\left(2\sqrt{\frac{2}{\pi}}\left(z + 0.044715\, z^3\right)\right)` # noqa: E501

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/SiELU.png

    Examples::

        >>> m = torch_activation.SiELU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SiELU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        inner = 2 * math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
        return x * torch.sigmoid(inner)
