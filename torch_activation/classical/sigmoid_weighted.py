import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_activation.base import BaseActivation
import math

from torch import Tensor
from torch_activation.utils import sech
from torch_activation import register_activation


# TODO: There are mentioned of WiG - a gated unit. Investigate it later..
@register_activation
class SiLU(BaseActivation):
    r"""
    Applies the Sigmoid Linear Unit activation function:

    :math:`\text{SiLU}(x) = x \cdot \sigma(x)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return F.silu(x)


@register_activation
class CoLU(BaseActivation):
    r"""
    Applies the Collapsing Linear Unit activation function:

    :math:`\text{CoLU}(x) = \frac{x}{1-x \cdot e^{-(x + e^x)}}`

     See: https://doi.org/10.48550/arXiv.2112.12078

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CoLU.png

    Examples::

        >>> m = nn.CoLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.CoLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        if self.inplace:
            return x.div_(1 - x * torch.exp(-1 * (x + torch.exp(x))))
        else:
            return x / (1 - x * torch.exp(-1 * (x + torch.exp(x))))


@register_activation
class Phish(torch.nn.Module):
    r"""
    Applies the Phish activation function:

    :math:`\text{Phish}(x) = x \cdot \tanh (\text{GELU} (x))`

     See: `Phish: A Novel Hyper-Optimizable Activation Function`_.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Phish.png

    Examples:
        >>> m = Phish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

    .. _`Phish: A Novel Hyper-Optimizable Activation Function`:
        https://www.semanticscholar.org/paper/Phish%3A-A-Novel-Hyper-Optimizable-Activation-Naveen/43eb5e22da6092d28f0e842fec53ec1a76e1ba6b
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        output = F.gelu(x)
        output = F.tanh(output)
        output = x * output
        return output


@register_activation
class SinLU(BaseActivation):
    r"""
    Applies the Sinu-sigmoidal Linear Unit activation function:

    :math:`\text{SinLU}(x) = (x + a \cdot \sin (b \cdot x)) \sigma (x)`

     See: https://doi.org/10.3390/math10030337

    Args:
        a (float, optional): Initial value for sine function magnitude. Default: 1.0.
        b (float, optional): Initial value for sine function period. Default: 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SinLU.png

    Examples::

        >>> m = nn.SinLU(a=5.0, b=6.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(Tensor([a]))
        self.beta = nn.Parameter(Tensor([b]))

    def _forward(self, x):
        result = x + self.alpha * torch.sin(self.beta * x)
        result *= torch.sigmoid(x)
        return result

    def _forward_inplace(self, x):
        s_x = torch.sigmoid(x)
        x.add_(self.alpha * torch.sin(self.beta * x))
        x.mul_(s_x)
        return x


@register_activation
class GELU(BaseActivation):
    r"""
    Applies the Gaussian Error Linear Unit activation function:

    :math:`\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2} \left( 1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right) \right)`

    This is a wrapper around PyTorch's native F.gelu implementation.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = GaussianErrorLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return F.gelu(x)


@register_activation
class SGELU(BaseActivation):
    r"""
    Applies the Symmetrical Gaussian Error Linear Unit activation function:

    :math:`\text{SGELU}(z) = a \cdot z \cdot \text{erf}\left(\frac{z}{\sqrt{2}}\right)`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SGELU(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]), requires_grad=False)

    def _forward(self, x) -> Tensor:
        return self.a * x * torch.erf(x / math.sqrt(2))


@register_activation
class CaLU(BaseActivation):
    r"""
    Applies the Cauchy Linear Unit activation function:

    :math:`\text{CaLU}(z) = z \cdot \Phi_{\text{Cauchy}}(z) = z \cdot \left( \frac{\arctan(z)}{\pi} + \frac{1}{2} \right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = CaLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * (torch.arctan(x) / math.pi + 0.5)


@register_activation
class LaLU(BaseActivation):
    r"""
    Applies the Laplace Linear Unit activation function:

    :math:`\text{LaLU}(z) = z \cdot \Phi_{\text{Laplace}}(z) = z \cdot \begin{cases} 
    1 - \frac{1}{2} \exp(-z), & z \geq 0 \\ 
    \frac{1}{2} \exp(z), & z < 0 
    \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LaLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
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

    Examples::

        >>> m = CoLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = CoLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        denominator = 1 - x * torch.exp(-(x + torch.exp(x)))
        return x.div_(denominator) if self.inplace else x / denominator


@register_activation
class TSSwish(BaseActivation):
    r"""
    Applies the Triple State Swish activation function:

    :math:`\text{TSS}(z) = z \cdot \frac{1}{1 + \exp(-z)} \left( \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z+a)} + \frac{1}{1 + \exp(-z+b)} \right)`

    Args:
        a (float, optional): First shift parameter. Default: 1.0
        b (float, optional): Second shift parameter. Default: 2.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TSSwish(a=1.5, b=2.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def _forward(self, x) -> Tensor:
        # TODO: Memory
        sigmoid_x = torch.sigmoid(x)
        triple_term = sigmoid_x + torch.sigmoid(x - self.a) + torch.sigmoid(x - self.b)
        return x * sigmoid_x * triple_term


@register_activation
class GSwish(BaseActivation):
    r"""
    Applies the Generalized Swish activation function:

    :math:`\text{GSwish}(z) = z \cdot \sigma(\exp(-z))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = GSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.sigmoid(torch.exp(-x))


@register_activation
class ESwish(BaseActivation):
    r"""
    Applies the Exponential Swish activation function:

    :math:`\text{ESwish}(z) = \exp(-z) \cdot \sigma(z)`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ESwish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return torch.exp(-x) * torch.sigmoid(x)


@register_activation
class dSigmoid(BaseActivation):
    r"""
    Applies the Derivative of Sigmoid Function activation:

    :math:`\text{dSigmoid}(z) = \exp(-z) \cdot (\sigma(z))^2`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = dSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return torch.exp(-x) * sigmoid_x * sigmoid_x


@register_activation
class Gish(BaseActivation):
    r"""
    Applies the Gish activation function:

    :math:`\text{Gish}(z) = z \cdot \ln(2 - \exp(-\exp(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Gish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.log(2 - torch.exp(-torch.exp(x)))


@register_activation
class Logish(BaseActivation):
    r"""
    Applies the Logish activation function:

    :math:`\text{Logish}(z) = z \cdot \ln(1 + \sigma(z))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Logish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.log(1 + torch.sigmoid(x))


@register_activation
class LogLogish(BaseActivation):
    r"""
    Applies the LogLogish activation function:

    :math:`\text{LogLogish}(z) = z \cdot (1 - \exp(-\exp(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LogLogish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * (1 - torch.exp(-torch.exp(x)))


@register_activation
class ExpExpish(BaseActivation):
    r"""
    Applies the ExpExpish activation function:

    :math:`\text{ExpExpish}(z) = z \cdot \exp(-\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ExpExpish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.exp(-torch.exp(-x))


@register_activation
class SelfArctan(BaseActivation):
    r"""
    Applies the SelfArctan activation function:

    :math:`\text{SelfArctan}(z) = z \cdot \arctan(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SelfArctan()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.arctan(x)


@register_activation
class pLogish(BaseActivation):
    r"""
    Applies the Parametric Logish activation function:

    :math:`\text{pLogish}(z_i) = a \cdot z_i \cdot \ln(1 + \sigma(b \cdot z_i))`

    where :math:`\sigma` is the sigmoid function.

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Sigmoid scale parameter. Default: 10.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = pLogish(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]), requires_grad=False)
        self.b = nn.Parameter(Tensor([b]), requires_grad=False)

    def _forward(self, x) -> Tensor:
        return self.a * x * torch.log(1 + torch.sigmoid(self.b * x))


@register_activation
class Phish(BaseActivation):
    r"""
    Applies the Phish activation function:

    :math:`\text{Phish}(z) = z \cdot \tanh(\text{GELU}(z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Phish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.tanh(F.gelu(x))


@register_activation
class Suish(BaseActivation):
    r"""
    Applies the Suish activation function:

    :math:`\text{Suish}(z) = \max(z, z \cdot \exp(-|z|))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Suish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return torch.maximum(x, x * torch.exp(-torch.abs(x)))


@register_activation
class TSReLU(BaseActivation):
    r"""
    Applies the Tangent Sigmoid ReLU activation function:

    :math:`\text{TSReLU}(z) = z \cdot \tanh(\sigma(z))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TSReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.tanh(torch.sigmoid(x))


@register_activation
class TBSReLU(BaseActivation):
    r"""
    Applies the Tangent Bipolar Sigmoid ReLU activation function:

    :math:`\text{TBSReLU}(z) = z \cdot \tanh\left(\frac{1 - \exp(-z)}{1 + \exp(-z)}\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TBSReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        exp_neg_x = torch.exp(-x)
        bipolar_sigmoid = (1 - exp_neg_x) / (1 + exp_neg_x)
        return x * torch.tanh(bipolar_sigmoid)


@register_activation
class LogSigmoid(BaseActivation):
    r"""
    Applies the LogSigmoid activation function:

    :math:`\text{LogSigmoid}(z) = \ln(\sigma(z)) = \ln\left(\frac{1}{1 + \exp(-z)}\right)`

    This is a wrapper around PyTorch's native F.logsigmoid implementation.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LogSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return F.logsigmoid(x)


@register_activation
class dSiLU(BaseActivation):
    r"""
    Applies the Derivative of SiLU activation function:

    :math:`\text{dSiLU}(z) = \sigma(z) \cdot (1 + z \cdot (1 - \sigma(z)))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = dSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return sigmoid_x * (1 + x * (1 - sigmoid_x))


@register_activation
class DoubleSiLU(BaseActivation):
    r"""
    Applies the Double SiLU activation function:

    :math:`\text{DoubleSiLU}(z) = z \cdot \frac{1}{1 + \exp\left(-z \cdot \frac{1}{1 + \exp(-z)}\right)}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = DoubleSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return F.silu(F.silu(x))


@register_activation
class MSiLU(BaseActivation):
    r"""
    Applies the Modified SiLU activation function:

    :math:`\text{MSiLU}(z) = z \cdot \sigma(z) + \exp\left(-\frac{z^2}{4}\right)`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.sigmoid(x) + torch.exp((-x.pow(2) - 1) / 4)


@register_activation
class TSiLU(BaseActivation):
    r"""
    Applies the Hyperbolic Tangent Sigmoid-Weighted Linear Unit activation function:

    :math:`\text{TSiLU}(z) = \frac{\exp\left(\frac{z}{1 + \exp(-z)}\right) - \exp\left(-\frac{z}{1 + \exp(-z)}\right)}{\exp\left(\frac{z}{1 + \exp(-z)}\right) + \exp\left(\frac{z}{1 + \exp(-z)}\right)}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        # The paper wrote this in tanh exp form
        silu_x = x * torch.sigmoid(x)
        return torch.tanh(silu_x)


@register_activation
class ASiLU(BaseActivation):
    r"""
    Applies the Arctan SiLU activation function:

    :math:`\text{ASiLU}(z) = \arctan(z) \cdot \frac{1}{1 + \exp(-z)}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ASiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return torch.arctan(x * torch.sigmoid(x))


# FIXME: Notation in paper is not clear, requires verification
# Should be done in a few days. Contact author when done.
@register_activation
class SwAT(BaseActivation):
    r"""
    Applies the SwAT activation function:

    :math:`\text{SwAT}(z) = z \cdot \frac{1}{1 + \exp(-\arctan(z))}`

    See: `https://drive.google.com/file/d/10g-lrsc4WhxU90zQLaBY9BuYconaj-vD/view?usp=sharing`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SwAT()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.sigmoid(torch.arctan(x))


@register_activation
class ReHSec(BaseActivation):
    r"""
    Applies the Rectified Hyperbolic Secant activation function:

    :math:`\text{ReHSec}(z) = z \cdot \text{sech}(z)`

    where sech is the hyperbolic secant function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ReHSec()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * sech(x)


@register_activation
class LiSHT(BaseActivation):
    r"""
    Applies the Linearly Scaled Hyperbolic Tangent activation function:

    :math:`\text{LiSHT}(z) = z \cdot \tanh(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LiSHT()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.tanh(x)


@register_activation
class Mish(BaseActivation):
    r"""
    Applies the Mish activation function:

    :math:`\text{Mish}(z) = z \cdot \tanh(\text{softplus}(z)) = z \cdot \tanh(\ln(1 + \exp(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.tanh(F.softplus(x))


@register_activation
class Smish(BaseActivation):
    r"""
    Applies the Smish activation function:

    :math:`\text{Smish}(z) = a \cdot z \cdot \tanh(\ln(1 + \sigma(b \cdot z)))`

    where :math:`\sigma` is the sigmoid function.

    :note: a = 1.0, b = 1.0 is the recommended through a parameter search. [need citation]

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Sigmoid scale parameter. Default: 1.0
        learnable (bool, optional): If True, the parameters are learnable. Default: False

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Smish(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(
        self, a: float = 1.0, b: float = 1.0, learnable: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

        if not learnable:
            self.a.requires_grad = False
            self.b.requires_grad = False

    def _forward(self, x) -> Tensor:
        return self.a * x * torch.tanh(torch.log(1 + torch.sigmoid(self.b * x)))


@register_activation
class TanhExp(BaseActivation):
    r"""
    Applies the TanhExp activation function:

    :math:`\text{TanhExp}(z) = z \cdot \tanh(\exp(z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TanhExp()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.tanh(torch.exp(x))


@register_activation
class Serf(BaseActivation):
    r"""
    Applies the Serf activation function:

    :math:`\text{Serf}(z) = z \cdot \text{erf}(\ln(1 + \exp(z)))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Serf()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
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

    Examples::

        >>> m = EANAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        exp_x = torch.exp(x)
        return x * (exp_x / (exp_x + 2))


@register_activation
class SinSig(BaseActivation):
    r"""
    Applies the SinSig activation function:

    :math:`\text{SinSig}(z) = z \cdot \sin\left(\frac{\pi}{2} \sigma(z)\right)`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SinSig()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        return x * torch.sin((math.pi / 2) * torch.sigmoid(x))


@register_activation
class SiELU(BaseActivation):
    r"""
    Applies the Gaussian Error Linear Unit with Sigmoid Activation Functions:

    :math:`\text{SiELU}(z) = z \cdot \sigma\left(\sqrt{\frac{2}{\pi}} z + 0.044715 z^3\right)`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SiELU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x) -> Tensor:
        inner = 2 * math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
        return x * torch.sigmoid(inner)
