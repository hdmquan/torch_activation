import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from torch_activation import register_activation


@register_activation
class CoLU(nn.Module):
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

    def __init__(self, inplace=False):
        super(CoLU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
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

    def __init__(self):
        super(Phish, self).__init__()

    def forward(self, x) -> Tensor:
        output = F.gelu(x)
        output = F.tanh(output)
        output = x * output
        return output


@register_activation
class SinLU(nn.Module):
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

    def __init__(self, a: float = 1.0, b: float = 1.0, inplace: bool = False):
        super(SinLU, self).__init__()
        self.alpha = nn.Parameter(Tensor([a]))
        self.beta = nn.Parameter(Tensor([b]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

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
class GaussianErrorLinearUnit(nn.Module):
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

    def __init__(self):
        super(GaussianErrorLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        return F.gelu(x)


@register_activation
class SymmetricalGaussianErrorLinearUnit(nn.Module):
    r"""
    Applies the Symmetrical Gaussian Error Linear Unit activation function:

    :math:`\text{SGELU}(z) = a \cdot z \cdot \text{erf}\left(\frac{z}{\sqrt{2}}\right)`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SymmetricalGaussianErrorLinearUnit(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(SymmetricalGaussianErrorLinearUnit, self).__init__()
        self.a = nn.Parameter(Tensor([a]))

    def forward(self, x) -> Tensor:
        return self.a * x * torch.erf(x / math.sqrt(2))


@register_activation
class CauchyLinearUnit(nn.Module):
    r"""
    Applies the Cauchy Linear Unit activation function:

    :math:`\text{CaLU}(z) = z \cdot \Phi_{\text{Cauchy}}(z) = z \cdot \left( \frac{\arctan(z)}{\pi} + \frac{1}{2} \right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = CauchyLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(CauchyLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        return x * (torch.arctan(x) / math.pi + 0.5)


@register_activation
class LaplaceLinearUnit(nn.Module):
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

        >>> m = LaplaceLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(LaplaceLinearUnit, self).__init__()

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        result[pos_mask] = x[pos_mask] * (1 - 0.5 * torch.exp(-x[pos_mask]))
        result[neg_mask] = x[neg_mask] * (0.5 * torch.exp(x[neg_mask]))
        
        return result


@register_activation
class CollapsingLinearUnit(nn.Module):
    r"""
    Applies the Collapsing Linear Unit activation function:

    :math:`\text{CoLU}(z) = z \cdot \frac{1}{1 - z \exp(-(z + \exp(z)))}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = CollapsingLinearUnit()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = CollapsingLinearUnit(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False):
        super(CollapsingLinearUnit, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            denom = 1 - x * torch.exp(-(x + torch.exp(x)))
            return x.div_(denom)
        else:
            denom = 1 - x * torch.exp(-(x + torch.exp(x)))
            return x / denom


@register_activation
class TripleStateSwish(nn.Module):
    r"""
    Applies the Triple State Swish activation function:

    :math:`\text{TSS}(z) = z \cdot \frac{1}{1 + \exp(-z)} \left( \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z+a)} + \frac{1}{1 + \exp(-z+b)} \right)`

    Args:
        a (float, optional): First shift parameter. Default: 1.0
        b (float, optional): Second shift parameter. Default: 2.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TripleStateSwish(a=1.5, b=2.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, inplace: bool = False):
        super(TripleStateSwish, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        sigmoid_x = torch.sigmoid(x)
        triple_term = sigmoid_x + torch.sigmoid(x - self.a) + torch.sigmoid(x - self.b)
        return x * sigmoid_x * triple_term

    def _forward_inplace(self, x):
        sigmoid_x = torch.sigmoid(x)
        triple_term = sigmoid_x + torch.sigmoid(x - self.a) + torch.sigmoid(x - self.b)
        x.mul_(sigmoid_x * triple_term)
        return x


@register_activation
class GeneralizedSwish(nn.Module):
    r"""
    Applies the Generalized Swish activation function:

    :math:`\text{GSwish}(z) = z \cdot \sigma(\exp(-z))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = GeneralizedSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(GeneralizedSwish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.sigmoid(torch.exp(-x))


@register_activation
class ExponentialSwish(nn.Module):
    r"""
    Applies the Exponential Swish activation function:

    :math:`\text{ExponentialSwish}(z) = \exp(-z) \cdot \sigma(z)`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ExponentialSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ExponentialSwish, self).__init__()

    def forward(self, x) -> Tensor:
        return torch.exp(-x) * torch.sigmoid(x)


@register_activation
class DerivativeOfSigmoidFunction(nn.Module):
    r"""
    Applies the Derivative of Sigmoid Function activation:

    :math:`\text{DerivativeOfSigmoidFunction}(z) = \exp(-z) \cdot (\sigma(z))^2`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = DerivativeOfSigmoidFunction()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(DerivativeOfSigmoidFunction, self).__init__()

    def forward(self, x) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return torch.exp(-x) * sigmoid_x * sigmoid_x


@register_activation
class Gish(nn.Module):
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

    def __init__(self):
        super(Gish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.log(2 - torch.exp(-torch.exp(x)))


@register_activation
class Logish(nn.Module):
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

    def __init__(self):
        super(Logish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.log(1 + torch.sigmoid(x))


@register_activation
class LogLogish(nn.Module):
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

    def __init__(self):
        super(LogLogish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * (1 - torch.exp(-torch.exp(x)))


@register_activation
class ExpExpish(nn.Module):
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

    def __init__(self):
        super(ExpExpish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.exp(-torch.exp(-x))


@register_activation
class SelfArctan(nn.Module):
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

    def __init__(self):
        super(SelfArctan, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.arctan(x)


@register_activation
class ParametricLogish(nn.Module):
    r"""
    Applies the Parametric Logish activation function:

    :math:`\text{ParametricLogish}(z_i) = a \cdot z_i \cdot \ln(1 + \sigma(b \cdot z_i))`

    where :math:`\sigma` is the sigmoid function.

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Sigmoid scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ParametricLogish(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(ParametricLogish, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def forward(self, x) -> Tensor:
        return self.a * x * torch.log(1 + torch.sigmoid(self.b * x))


@register_activation
class Phish(nn.Module):
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

    def __init__(self):
        super(Phish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.tanh(F.gelu(x))


@register_activation
class Suish(nn.Module):
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

    def __init__(self):
        super(Suish, self).__init__()

    def forward(self, x) -> Tensor:
        return torch.maximum(x, x * torch.exp(-torch.abs(x)))


@register_activation
class TangentSigmoidReLU(nn.Module):
    r"""
    Applies the Tangent Sigmoid ReLU activation function:

    :math:`\text{TangentSigmoidReLU}(z) = z \cdot \tanh(\sigma(z))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TangentSigmoidReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(TangentSigmoidReLU, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.tanh(torch.sigmoid(x))


@register_activation
class TangentBipolarSigmoidReLU(nn.Module):
    r"""
    Applies the Tangent Bipolar Sigmoid ReLU activation function:

    :math:`\text{TangentBipolarSigmoidReLU}(z) = z \cdot \tanh\left(\frac{1 - \exp(-z)}{1 + \exp(-z)}\right)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TangentBipolarSigmoidReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(TangentBipolarSigmoidReLU, self).__init__()

    def forward(self, x) -> Tensor:
        exp_neg_x = torch.exp(-x)
        bipolar_sigmoid = (1 - exp_neg_x) / (1 + exp_neg_x)
        return x * torch.tanh(bipolar_sigmoid)


@register_activation
class LogSigmoid(nn.Module):
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

    def __init__(self):
        super(LogSigmoid, self).__init__()

    def forward(self, x) -> Tensor:
        return F.logsigmoid(x)


@register_activation
class DerivativeOfSiLU(nn.Module):
    r"""
    Applies the Derivative of SiLU activation function:

    :math:`\text{DerivativeOfSiLU}(z) = \sigma(z) \cdot (1 + z \cdot (1 - \sigma(z)))`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = DerivativeOfSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(DerivativeOfSiLU, self).__init__()

    def forward(self, x) -> Tensor:
        sigmoid_x = torch.sigmoid(x)
        return sigmoid_x * (1 + x * (1 - sigmoid_x))


@register_activation
class DoubleSiLU(nn.Module):
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

    def __init__(self):
        super(DoubleSiLU, self).__init__()

    def forward(self, x) -> Tensor:
        silu_x = x * torch.sigmoid(x)
        return x * torch.sigmoid(silu_x)


@register_activation
class ModifiedSiLU(nn.Module):
    r"""
    Applies the Modified SiLU activation function:

    :math:`\text{ModifiedSiLU}(z) = z \cdot \sigma(z) + \exp\left(-\frac{z^2}{4}\right)`

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ModifiedSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ModifiedSiLU, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.sigmoid(x) + torch.exp(-x.pow(2) / 4)


@register_activation
class HyperbolicTangentSiLU(nn.Module):
    r"""
    Applies the Hyperbolic Tangent SiLU activation function:

    :math:`\text{HyperbolicTangentSiLU}(z) = \frac{\exp\left(\frac{z}{1 + \exp(-z)}\right) - \exp\left(-\frac{z}{1 + \exp(-z)}\right)}{\exp\left(\frac{z}{1 + \exp(-z)}\right) + \exp\left(\frac{z}{1 + \exp(-z)}\right)}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HyperbolicTangentSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(HyperbolicTangentSiLU, self).__init__()

    def forward(self, x) -> Tensor:
        silu_x = x * torch.sigmoid(x)
        return torch.tanh(silu_x)

@register_activation
class ArctanSiLU(nn.Module):
    r"""
    Applies the Arctan SiLU activation function:

    :math:`\text{ArctanSiLU}(z) = \arctan(z) \cdot \frac{1}{1 + \exp(-z)}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ArctanSiLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ArctanSiLU, self).__init__()

    def forward(self, x) -> Tensor:
        return torch.arctan(x) * torch.sigmoid(x)


@register_activation
class SwAT(nn.Module):
    r"""
    Applies the SwAT activation function:

    :math:`\text{SwAT}(z) = z \cdot \frac{1}{1 + \exp(-\arctan(|z|))}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SwAT()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SwAT, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.sigmoid(torch.arctan(torch.abs(x)))


@register_activation
class RectifiedHyperbolicSecant(nn.Module):
    r"""
    Applies the Rectified Hyperbolic Secant activation function:

    :math:`\text{RectifiedHyperbolicSecant}(z) = z \cdot \text{sech}(z)`

    where sech is the hyperbolic secant function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = RectifiedHyperbolicSecant()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(RectifiedHyperbolicSecant, self).__init__()

    def forward(self, x) -> Tensor:
        return x * (2 / (torch.exp(x) + torch.exp(-x)))


@register_activation
class LinearlyScaledHyperbolicTangent(nn.Module):
    r"""
    Applies the Linearly Scaled Hyperbolic Tangent activation function:

    :math:`\text{LinearlyScaledHyperbolicTangent}(z) = z \cdot \tanh(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LinearlyScaledHyperbolicTangent()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(LinearlyScaledHyperbolicTangent, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.tanh(x)


@register_activation
class Mish(nn.Module):
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

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.tanh(F.softplus(x))


@register_activation
class Smish(nn.Module):
    r"""
    Applies the Smish activation function:

    :math:`\text{Smish}(z) = a \cdot z \cdot \tanh(\ln(1 + \sigma(b \cdot z)))`

    where :math:`\sigma` is the sigmoid function.

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Sigmoid scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Smish(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(Smish, self).__init__()
        self.a = nn.Parameter(Tensor([a]))
        self.b = nn.Parameter(Tensor([b]))

    def forward(self, x) -> Tensor:
        return self.a * x * torch.tanh(torch.log(1 + torch.sigmoid(self.b * x)))


@register_activation
class TanhExp(nn.Module):
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

    def __init__(self):
        super(TanhExp, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.tanh(torch.exp(x))


@register_activation
class Serf(nn.Module):
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

    def __init__(self):
        super(Serf, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.erf(F.softplus(x))


@register_activation
class EfficientAsymmetricNonlinearActivationFunction(nn.Module):
    r"""
    Applies the Efficient Asymmetric Nonlinear Activation Function:

    :math:`\text{EfficientAsymmetricNonlinearActivationFunction}(z) = z \cdot \frac{\exp(z)}{\exp(z) + 2}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = EfficientAsymmetricNonlinearActivationFunction()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(EfficientAsymmetricNonlinearActivationFunction, self).__init__()

    def forward(self, x) -> Tensor:
        exp_x = torch.exp(x)
        return x * (exp_x / (exp_x + 2))


@register_activation
class SinSig(nn.Module):
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

    def __init__(self):
        super(SinSig, self).__init__()

    def forward(self, x) -> Tensor:
        return x * torch.sin((math.pi / 2) * torch.sigmoid(x))


@register_activation
class SiELU(nn.Module):
    r"""
    Applies the SiELU activation function:

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

    def __init__(self):
        super(SiELU, self).__init__()

    def forward(self, x) -> Tensor:
        inner = math.sqrt(2/math.pi) * x + 0.044715 * x.pow(3)
        return x * torch.sigmoid(inner)