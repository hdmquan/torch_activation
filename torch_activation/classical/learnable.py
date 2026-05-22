import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class TAAF(BaseActivation):
    r"""
    Applies the Transformative Adaptive Activation Function (TAAF):

    :math:`\text{TAAF}(x) = \alpha \cdot f(\beta \cdot x + \gamma) + \delta`

    where :math:`f` is tanh and :math:`\alpha, \beta, \gamma, \delta` are learnable parameters.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TAAF.png

    Examples::

        >>> m = torch_activation.TAAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(1))

    def _forward(self, x) -> Tensor:
        return self.alpha * torch.tanh(self.beta * x + self.gamma) + self.delta


@register_activation
class tSoftmax(BaseActivation):
    r"""
    Applies the temperature-scaled Softmax function:

    :math:`\text{tSoftmax}(x_i) = \frac{\exp(x_i / t)}{\sum_j \exp(x_j / t)}`

    where :math:`t` is a learnable temperature parameter.

    Args:
        dim (int, optional): dimension along which softmax is applied. Default: ``-1``
        init_t (float, optional): initial temperature value. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/tSoftmax.png

    Examples::

        >>> m = torch_activation.tSoftmax()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, init_t: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.t = nn.Parameter(torch.tensor(init_t))

    def _forward(self, x) -> Tensor:
        t = self.t.abs().clamp(min=1e-6)
        return F.softmax(x / t, dim=self.dim)


@register_activation
class GEU(BaseActivation):
    r"""
    Applies the Gaussian Error Unit (GEU) activation function:

    :math:`\text{GEU}(x) = x \cdot \Phi(\alpha \cdot x)`

    where :math:`\Phi` is the standard Gaussian CDF and :math:`\alpha` is a learnable scaling parameter.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GEU.png

    Examples::

        >>> m = torch_activation.GEU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.ones(1))

    def _forward(self, x) -> Tensor:
        return x * 0.5 * (1.0 + torch.erf(self.alpha * x / math.sqrt(2.0)))


@register_activation
class SAAAF(BaseActivation):
    r"""
    Applies the Smooth Adaptive Activation Function (SAAAF):

    :math:`\text{SAAAF}(x) = \sum_{k} w_k \cdot b_k(x)`

    where :math:`b_k(x)` are piecewise polynomial basis functions (hat functions) and :math:`w_k` are learnable weights.

    Args:
        n_segments (int, optional): number of piecewise segments. Default: ``8``
        x_min (float, optional): lower bound of the input range. Default: ``-4.0``
        x_max (float, optional): upper bound of the input range. Default: ``4.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SAAAF.png

    Examples::

        >>> m = torch_activation.SAAAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_segments: int = 8, x_min: float = -4.0, x_max: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.n_segments = n_segments
        self.x_min = x_min
        self.x_max = x_max
        self.weights = nn.Parameter(torch.linspace(x_min, x_max, n_segments + 1))

    def _forward(self, x) -> Tensor:
        knots = torch.linspace(
            self.x_min, self.x_max, self.n_segments + 1, device=x.device, dtype=x.dtype
        )
        delta = knots[1] - knots[0]
        result = torch.zeros_like(x)
        for k in range(self.n_segments + 1):
            hat = (1.0 - (x - knots[k]).abs() / delta).clamp(min=0.0)
            result = result + self.weights[k] * hat
        return result


@register_activation
class ScaledSoftsign(BaseActivation):
    r"""
    Applies the Scaled Softsign activation function:

    :math:`\text{ScaledSoftsign}(x) = \frac{\alpha \cdot x}{1 + |x|}`

    where :math:`\alpha` is a learnable scaling parameter.

    Args:
        alpha (float, optional): initial value of scaling parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledSoftsign.png

    Examples::

        >>> m = torch_activation.ScaledSoftsign()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def _forward(self, x) -> Tensor:
        return self.alpha * x / (1.0 + x.abs())


@register_activation
class pSoftplus(BaseActivation):
    r"""
    Applies the parametric Softplus activation function:

    :math:`\text{pSoftplus}(x) = \frac{1}{\beta} \ln(1 + \exp(\beta \cdot x))`

    where :math:`\beta` is a learnable parameter controlling the sharpness.

    Args:
        beta (float, optional): initial value of learnable sharpness parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/pSoftplus.png

    Examples::

        >>> m = torch_activation.pSoftplus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))

    def _forward(self, x) -> Tensor:
        beta = self.log_beta.exp().clamp(min=1e-6)
        return F.softplus(beta * x) / beta


@register_activation
class LEAF(BaseActivation):
    r"""
    Applies the Learnable Extended Activation Function (LEAF):

    :math:`\text{LEAF}(x) = \alpha \cdot \max(x, 0) + \beta \cdot \min(x, 0) \cdot \sigma(\gamma \cdot x)`

    where :math:`\alpha, \beta, \gamma` are learnable parameters.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LEAF.png

    Examples::

        >>> m = torch_activation.LEAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

    def _forward(self, x) -> Tensor:
        pos = self.alpha * F.relu(x)
        neg = self.beta * (-F.relu(-x)) * torch.sigmoid(self.gamma * x)
        return pos + neg


@register_activation
class ELUpSoftplus(BaseActivation):
    r"""
    Applies the ELU-Softplus activation function:

    .. math::

        \text{ELUpSoftplus}(x) = \begin{cases}
        x, & x \geq 0 \\
        \alpha(\exp(x) - 1) + \ln(1 + \exp(\beta \cdot x)), & x < 0
        \end{cases}

    Args:
        alpha (float, optional): ELU scale for negative inputs. Default: ``1.0``
        beta (float, optional): Softplus sharpness for negative inputs. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ELUpSoftplus.png

    Examples::

        >>> m = torch_activation.ELUpSoftplus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def _forward(self, x) -> Tensor:
        elu_neg = self.alpha * (x.exp() - 1.0)
        sp_neg = F.softplus(self.beta * x)
        neg_part = elu_neg + sp_neg
        return torch.where(x >= 0, x, neg_part)


@register_activation
class APLU(BaseActivation):
    r"""
    Applies the Adaptive Piecewise Linear Unit (APLU):

    :math:`\text{APLU}(x) = \max(0, x) + \sum_{s=1}^{S} a^s \max(0, -x + b^s)`

    where :math:`a^s` and :math:`b^s` are learnable parameters for each hinge.

    Args:
        n_hinges (int, optional): number of hinge functions S. Default: ``2``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/APLU.png

    Examples::

        >>> m = torch_activation.APLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_hinges: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.zeros(n_hinges))
        self.b = nn.Parameter(torch.linspace(-1.0, 1.0, n_hinges))

    def _forward(self, x) -> Tensor:
        out = F.relu(x)
        for s in range(self.a.shape[0]):
            out = out + self.a[s] * F.relu(-x + self.b[s])
        return out


@register_activation
class SPLASH(BaseActivation):
    r"""
    Applies the Symmetric Piecewise Linear Adaptive Squashing Hinge (SPLASH) unit:

    :math:`\text{SPLASH}(x) = \sum_{s=1}^{S} a_s \cdot h(x; c_s)`

    where :math:`h(x; c) = \max(0, x - c) + \max(0, -x - c)` is a symmetric hinge and :math:`c_s` are fixed hinge locations.

    Args:
        n_hinges (int, optional): number of symmetric hinges S. Default: ``4``
        hinge_range (float, optional): range for hinge location initialization. Default: ``2.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SPLASH.png

    Examples::

        >>> m = torch_activation.SPLASH()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_hinges: int = 4, hinge_range: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.ones(n_hinges))
        c = torch.linspace(0.0, hinge_range, n_hinges)
        self.register_buffer("c", c)

    def _forward(self, x) -> Tensor:
        out = torch.zeros_like(x)
        for s in range(self.a.shape[0]):
            hinge = F.relu(x - self.c[s]) + F.relu(-x - self.c[s])
            out = out + self.a[s] * hinge
        return out


@register_activation
class MBA(BaseActivation):
    r"""
    Applies the Mixture of Basis Activations (MBA):

    :math:`\text{MBA}(x) = \sum_{k=1}^{K} w_k \cdot f_k(x)`

    where :math:`f_k` are fixed basis activations (ReLU, tanh, sigmoid) and :math:`w_k` are learnable mixture weights.

    Args:
        n_bases (int, optional): number of basis functions. Default: ``3``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MBA.png

    Examples::

        >>> m = torch_activation.MBA()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_bases: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.weights = nn.Parameter(torch.ones(n_bases) / n_bases)

    _bases = [F.relu, torch.tanh, torch.sigmoid]

    def _forward(self, x) -> Tensor:
        n = min(self.weights.shape[0], len(self._bases))
        w = F.softmax(self.weights[:n], dim=0)
        out = torch.zeros_like(x)
        for k in range(n):
            out = out + w[k] * self._bases[k](x)
        return out


@register_activation
class AdaLU(BaseActivation):
    r"""
    Applies the Adaptive Linear Unit (AdaLU):

    .. math::

        \text{AdaLU}(x) = \begin{cases}
        \alpha \cdot x, & x \geq 0 \\
        \beta \cdot x, & x < 0
        \end{cases}

    where :math:`\alpha` and :math:`\beta` are learnable parameters.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/AdaLU.png

    Examples::

        >>> m = torch_activation.AdaLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.full((1,), 0.1))

    def _forward(self, x) -> Tensor:
        return torch.where(x >= 0, self.alpha * x, self.beta * x)


@register_activation
class TSAF(BaseActivation):
    r"""
    Applies the Temperature-Scaled Activation Function (TSAF):

    :math:`\text{TSAF}(x) = \tanh(t \cdot x)`

    where :math:`t` is a learnable temperature parameter.

    Args:
        init_t (float, optional): initial temperature value. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TSAF.png

    Examples::

        >>> m = torch_activation.TSAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, init_t: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.t = nn.Parameter(torch.tensor(init_t))

    def _forward(self, x) -> Tensor:
        return torch.tanh(self.t * x)


@register_activation
class ARiA(BaseActivation):
    r"""
    Applies the Adaptive Richard's Curve weighted Activation (ARiA):

    :math:`\text{ARiA}(x) = x \cdot \sigma(\beta \cdot x)^{1/\nu}`

    where :math:`\beta` controls the curve steepness and :math:`\nu` controls non-monotonicity.

    Args:
        beta (float, optional): steepness parameter. Default: ``1.0``
        nu (float, optional): non-monotonicity control. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ARiA.png

    Examples::

        >>> m = torch_activation.ARiA()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, beta: float = 1.0, nu: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.nu = nu

    def _forward(self, x) -> Tensor:
        return x * torch.sigmoid(self.beta * x) ** (1.0 / max(self.nu, 1e-6))


@register_activation
class MWF(BaseActivation):
    r"""
    Applies the Multi-Wavelet Function (MWF) activation:

    :math:`\text{MWF}(x) = a \cdot x \cdot \exp(-b \cdot x^2) + c \cdot \tanh(x)`

    where :math:`a, b, c` are learnable parameters.

    Args:
        a (float, optional): amplitude of wavelet component. Default: ``1.0``
        b (float, optional): width of wavelet component. Default: ``1.0``
        c (float, optional): weight of tanh component. Default: ``0.5``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MWF.png

    Examples::

        >>> m = torch_activation.MWF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))

    def _forward(self, x) -> Tensor:
        b = self.b.abs().clamp(min=1e-6)
        return self.a * x * torch.exp(-b * x**2) + self.c * torch.tanh(x)


@register_activation
class Sincos(BaseActivation):
    r"""
    Applies the Sincos activation function:

    :math:`\text{Sincos}(x) = a \sin(bx) + c \cos(dx)`

    where :math:`a`, :math:`b`, :math:`c`, :math:`d` are learnable parameters.

    Args:
        a (float, optional): Scale for sine term. Default: ``1.0``
        b (float, optional): Frequency for sine term. Default: ``1.0``
        c (float, optional): Scale for cosine term. Default: ``1.0``
        d (float, optional): Frequency for cosine term. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Sincos.png

    Examples::

        >>> m = torch_activation.Sincos()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Sincos(a=0.5, b=2.0, c=0.5, d=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))
        self.d = nn.Parameter(torch.tensor(d))

    def _forward(self, x) -> Tensor:
        return self.a * torch.sin(self.b * x) + self.c * torch.cos(self.d * x)


@register_activation
class CSS(BaseActivation):
    r"""
    Applies the Cosine-Sigmoid Shift (CSS) activation function:

    :math:`\text{CSS}(x) = a \sin(bx) + c \sigma(dx)`

    where :math:`\sigma` is the sigmoid function and :math:`a`, :math:`b`, :math:`c`, :math:`d`
    are learnable parameters.

    Args:
        a (float, optional): Scale for sine term. Default: ``1.0``
        b (float, optional): Frequency for sine term. Default: ``1.0``
        c (float, optional): Scale for sigmoid term. Default: ``1.0``
        d (float, optional): Scale for sigmoid input. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CSS.png

    Examples::

        >>> m = torch_activation.CSS()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.CSS(a=2.0, b=1.0, c=1.0, d=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))
        self.d = nn.Parameter(torch.tensor(d))

    def _forward(self, x) -> Tensor:
        return self.a * torch.sin(self.b * x) + self.c * torch.sigmoid(self.d * x)


@register_activation
class CatAF(BaseActivation):
    r"""
    Applies the Concatenation Activation Function (CatAF):

    :math:`\text{CatAF}(x) = \alpha \cdot \text{ReLU}(x) + \beta \cdot \tanh(x)`

    where :math:`\alpha` and :math:`\beta` are learnable weights.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CatAF.png

    Examples::

        >>> m = torch_activation.CatAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _forward(self, x) -> Tensor:
        return self.alpha * F.relu(x) + self.beta * torch.tanh(x)


@register_activation
class Expcos(BaseActivation):
    r"""
    Applies the Expcos activation function:

    :math:`\text{Expcos}(x) = \exp(-ax^2) \cos(bx)`

    where :math:`a` and :math:`b` are learnable parameters.

    Args:
        a (float, optional): Decay rate. Default: ``1.0``
        b (float, optional): Frequency. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Expcos.png

    Examples::

        >>> m = torch_activation.Expcos()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Expcos(a=0.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    def _forward(self, x) -> Tensor:
        return torch.exp(-self.a * x**2) * torch.cos(self.b * x)


@register_activation
class MTLU(BaseActivation):
    r"""
    Applies the Multi-bin Trainable Linear Unit (MTLU):

    :math:`\text{MTLU}(x) = \max(a_1, \min(a_2, x)) \cdot w_1 + \text{SReLU}(x) \cdot w_2`

    implemented as a sum of S linear segments with learnable slopes and breakpoints.

    Args:
        n_bins (int, optional): number of linear segments. Default: ``4``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MTLU.png

    Examples::

        >>> m = torch_activation.MTLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_bins: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.breakpoints = nn.Parameter(torch.linspace(-2.0, 2.0, n_bins - 1))
        self.slopes = nn.Parameter(torch.ones(n_bins))

    def _forward(self, x) -> Tensor:
        bp = self.breakpoints.sort().values
        out = self.slopes[0] * x
        for i, b in enumerate(bp):
            out = out + (self.slopes[i + 1] - self.slopes[i]) * F.relu(x - b)
        return out


@register_activation
class CPN(BaseActivation):
    r"""
    Applies the Cosine-Polynomial Network (CPN) activation:

    :math:`\text{CPN}(x) = \sum_{k=0}^{K} a_k \cos(k \cdot x)`

    where :math:`a_k` are learnable Fourier-like coefficients.

    Args:
        n_terms (int, optional): number of cosine terms K+1. Default: ``4``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CPN.png

    Examples::

        >>> m = torch_activation.CPN()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_terms: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.coeffs = nn.Parameter(torch.randn(n_terms) * 0.1)
        self.coeffs.data[0] = 0.0

    def _forward(self, x) -> Tensor:
        out = torch.zeros_like(x)
        for k in range(self.coeffs.shape[0]):
            out = out + self.coeffs[k] * torch.cos(k * x)
        return out


@register_activation
class LuTU(BaseActivation):
    r"""
    Applies the Lookup Table Unit (LuTU) activation:

    :math:`\text{LuTU}(x) = \sum_{k} v_k \cdot h\!\left(\frac{x - c_k}{\delta}\right)`

    where :math:`h(t) = 0.5(1 + \cos(\pi \cdot \text{clip}(t, -1, 1)))` is a cosine hat basis, :math:`c_k` are fixed anchor points and :math:`v_k` are learnable values.

    Args:
        n_anchors (int, optional): number of anchor points. Default: ``16``
        x_min (float, optional): minimum anchor. Default: ``-4.0``
        x_max (float, optional): maximum anchor. Default: ``4.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LuTU.png

    Examples::

        >>> m = torch_activation.LuTU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_anchors: int = 16, x_min: float = -4.0, x_max: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        anchors = torch.linspace(x_min, x_max, n_anchors)
        self.register_buffer("anchors", anchors)
        self.values = nn.Parameter(torch.linspace(x_min, x_max, n_anchors))

    def _forward(self, x) -> Tensor:
        delta = (self.anchors[-1] - self.anchors[0]) / (self.anchors.shape[0] - 1)
        out = torch.zeros_like(x)
        for k in range(self.anchors.shape[0]):
            t = (x - self.anchors[k]) / delta
            hat = 0.5 * (1.0 + torch.cos(math.pi * t.clamp(-1.0, 1.0)))
            out = out + self.values[k] * hat
        return out


@register_activation
class Maxout(BaseActivation):
    r"""
    Applies the Maxout activation function:

    :math:`\text{Maxout}(x) = \max_{k \in [1,K]} (w_k \cdot x + b_k)`

    where each neuron learns K affine functions and takes the maximum.

    Args:
        n_pieces (int, optional): number of linear pieces K. Default: ``2``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Maxout.png

    Examples::

        >>> m = torch_activation.Maxout()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, n_pieces: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.slopes = nn.Parameter(torch.randn(n_pieces))
        self.biases = nn.Parameter(torch.zeros(n_pieces))

    def _forward(self, x) -> Tensor:
        candidates = torch.stack(
            [self.slopes[k] * x + self.biases[k] for k in range(self.slopes.shape[0])], dim=0
        )
        return candidates.max(dim=0).values


@register_activation
class PAU(BaseActivation):
    r"""
    Applies the Padé Activation Unit (PAU):

    :math:`\text{PAU}(x) = \frac{\sum_{i=0}^{m} a_i x^i}{1 + \sum_{j=1}^{n} |b_j| x^{2j}}`

    where the denominator uses absolute values to avoid poles.

    Args:
        m (int, optional): numerator degree. Default: ``5``
        n (int, optional): denominator degree. Default: ``4``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PAU.png

    Examples::

        >>> m = torch_activation.PAU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, m: int = 5, n: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.n = n
        self.a = nn.Parameter(torch.zeros(m + 1))
        self.a.data[1] = 1.0
        self.b = nn.Parameter(torch.zeros(n))

    def extra_repr(self):
        return f"m={self.m}, n={self.n}"

    def _forward(self, x) -> Tensor:
        num = torch.zeros_like(x)
        for i, ai in enumerate(self.a):
            num = num + ai * x**i
        denom = torch.ones_like(x)
        for j, bj in enumerate(self.b):
            denom = denom + bj.abs() * x ** (2 * (j + 1))
        return num / denom


@register_activation
class RPAU(BaseActivation):
    r"""
    Applies the Robust Padé Activation Unit (RPAU):

    :math:`\text{RPAU}(x) = \frac{\sum_{i=0}^{m} a_i x^i}{1 + \sum_{j=1}^{n} |b_j| |x|^j}`

    Uses absolute values in the denominator for improved numerical stability.

    Args:
        m (int, optional): numerator degree. Default: ``5``
        n (int, optional): denominator degree. Default: ``4``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RPAU.png

    Examples::

        >>> m = torch_activation.RPAU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, m: int = 5, n: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.zeros(m + 1))
        self.a.data[1] = 1.0
        self.b = nn.Parameter(torch.zeros(n))

    def _forward(self, x) -> Tensor:
        num = torch.zeros_like(x)
        for i, ai in enumerate(self.a):
            num = num + ai * x**i
        ax = x.abs()
        denom = torch.ones_like(x)
        for j, bj in enumerate(self.b):
            denom = denom + bj.abs() * ax ** (j + 1)
        return num / denom


@register_activation
class ERA(BaseActivation):
    r"""
    Applies the Enhanced Rational Activation (ERA):

    :math:`\text{ERA}(x) = \frac{P(x)}{Q(x)}`

    where P and Q are learnable polynomials initialized to approximate ReLU, with the denominator constrained to be positive.

    Args:
        degree (int, optional): polynomial degree for numerator and denominator. Default: ``3``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ERA.png

    Examples::

        >>> m = torch_activation.ERA()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, degree: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.zeros(degree + 1))
        self.a.data[1] = 1.0
        self.b = nn.Parameter(torch.zeros(degree + 1))
        self.b.data[0] = 1.0

    def _forward(self, x) -> Tensor:
        num = torch.zeros_like(x)
        for i, ai in enumerate(self.a):
            num = num + ai * x**i
        denom = torch.zeros_like(x)
        for j, bj in enumerate(self.b):
            denom = denom + bj.abs() * x.abs() ** j
        denom = denom.clamp(min=1e-6)
        return num / denom


@register_activation
class OPAU(BaseActivation):
    r"""
    Applies the Orthogonal-Padé Activation Unit (OPAU):

    :math:`\text{OPAU}(x) = \frac{\sum_{i=0}^{k} c_i H_i(x)}{1 + \sum_{j=1}^{l} |d_j| |H_j(x)|}`

    where :math:`H_i` are Hermite polynomials as orthogonal basis functions.

    Args:
        k (int, optional): numerator degree. Default: ``3``
        denom_deg (int, optional): denominator degree. Default: ``2``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/OPAU.png

    Examples::

        >>> m = torch_activation.OPAU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, k: int = 3, denom_deg: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.c = nn.Parameter(torch.zeros(k + 1))
        self.c.data[1] = 1.0
        self.d = nn.Parameter(torch.zeros(denom_deg))

    @staticmethod
    def _hermite(x, n):
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            h_prev2, h_prev1 = torch.ones_like(x), x
            for i in range(2, n + 1):
                h_curr = x * h_prev1 - (i - 1) * h_prev2
                h_prev2, h_prev1 = h_prev1, h_curr
            return h_prev1

    def _forward(self, x) -> Tensor:
        num = torch.zeros_like(x)
        for i, ci in enumerate(self.c):
            num = num + ci * self._hermite(x, i)
        denom = torch.ones_like(x)
        for j, dj in enumerate(self.d):
            h = self._hermite(x, j + 1)
            denom = denom + dj.abs() * h.abs()
        return num / denom


@register_activation
class SAF(BaseActivation):
    r"""
    Applies the Self-Adaptable activation Function (SAF):

    :math:`\text{SAF}(x) = x \cdot \sigma(\alpha \cdot x + \beta)`

    where :math:`\alpha` and :math:`\beta` are learnable parameters, generalizing Swish.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SAF.png

    Examples::

        >>> m = torch_activation.SAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def _forward(self, x) -> Tensor:
        return x * torch.sigmoid(self.alpha * x + self.beta)


@register_activation
class TruG(BaseActivation):
    r"""
    Applies the True Gaussian activation function:

    :math:`\text{TruG}(x) = \exp\!\left(-\frac{x^2}{2\sigma^2}\right)`

    where :math:`\sigma` is a learnable width parameter.

    Args:
        sigma (float, optional): initial width parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TruG.png

    Examples::

        >>> m = torch_activation.TruG()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, sigma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))

    def _forward(self, x) -> Tensor:
        sigma2 = self.log_sigma.exp().pow(2).clamp(min=1e-6)
        return torch.exp(-(x**2) / (2.0 * sigma2))


@register_activation
class NIN(BaseActivation):
    r"""
    Applies the Network-in-Network (NIN) micro-MLP activation:

    :math:`\text{NIN}(x) = \text{ReLU}(w_2 \cdot \text{ReLU}(w_1 \cdot x + b_1) + b_2)`

    A two-layer MLP applied element-wise as an activation function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/NIN.png

    Examples::

        >>> m = torch_activation.NIN()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w1 = nn.Parameter(torch.randn(1) * 0.1 + 1.0)
        self.b1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.randn(1) * 0.1 + 1.0)
        self.b2 = nn.Parameter(torch.zeros(1))

    def _forward(self, x) -> Tensor:
        h = F.relu(self.w1 * x + self.b1)
        return F.relu(self.w2 * h + self.b2)


@register_activation
class SAVEBased(BaseActivation):
    r"""
    Applies the SAVE-based activation function:

    :math:`\text{SAVE}(x) = \alpha \cdot x + \beta \cdot \sin(x)`

    A combination of identity and sinusoidal components with learnable weights.

    Args:
        alpha (float, optional): initial weight for linear component. Default: ``1.0``
        beta (float, optional): initial weight for sinusoidal component. Default: ``0.5``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SAVEBased.png

    Examples::

        >>> m = torch_activation.SAVEBased()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def _forward(self, x) -> Tensor:
        return self.alpha * x + self.beta * torch.sin(x)


@register_activation
class GRA(BaseActivation):
    r"""
    Applies the Generalized Rational Activation (GRA):

    :math:`\text{GRA}(z_i) = 1 - \frac{a_i}{a_i + (1 + b_i z_i)^{c_i}}`

    where :math:`a_i, b_i, c_i` are learnable per-element parameters.

    Args:
        num_parameters (int, optional): number of learnable parameter sets. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GRA.png

    Examples::

        >>> m = torch_activation.GRA()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, num_parameters: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.ones(num_parameters))
        self.b = nn.Parameter(torch.ones(num_parameters))
        self.c = nn.Parameter(torch.full((num_parameters,), 2.0))

    def _forward(self, x) -> Tensor:
        a = self.a.abs().clamp(min=1e-6)
        return 1 - a / (a + (1 + self.b * x) ** self.c)


@register_activation
class EIS(BaseActivation):
    r"""
    Applies the EIS-3 activation function (Exponential Inverse Sigmoid):

    :math:`\text{EIS}(z_i) = \frac{z_i}{1 + d_i \exp(-e_i z_i)}`

    where :math:`d_i, e_i` are learnable per-element parameters.

    Args:
        num_parameters (int, optional): number of learnable parameter sets. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/EIS.png

    Examples::

        >>> m = torch_activation.EIS()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, num_parameters: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.d = nn.Parameter(torch.ones(num_parameters))
        self.e = nn.Parameter(torch.ones(num_parameters))

    def _forward(self, x) -> Tensor:
        return x / (1 + self.d * torch.exp(-self.e * x))


@register_activation
class ScaledLogisticSigmoid(BaseActivation):
    r"""
    Applies the Scaled Logistic Sigmoid activation:

    :math:`\text{ScaledLogisticSigmoid}(z_i) = \frac{a_i}{1 + \exp(-b_i z_i)}`

    where :math:`a_i, b_i` are learnable per-element parameters.

    Args:
        num_parameters (int, optional): number of learnable parameter sets. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledLogisticSigmoid.png

    Examples::

        >>> m = torch_activation.ScaledLogisticSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, num_parameters: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.ones(num_parameters))
        self.b = nn.Parameter(torch.ones(num_parameters))

    def _forward(self, x) -> Tensor:
        return self.a / (1 + torch.exp(-self.b * x))


@register_activation
class PLU(BaseActivation):
    r"""
    Applies the Piecewise Linear Unit (PLU):

    :math:`\text{PLU}(z_i) = \max(a_i(z_i + b) - b,\; \min(a_i(z_i - b) + b,\; z_i))`

    where :math:`a_i` are learnable parameters and :math:`b` is a fixed bound.

    Args:
        num_parameters (int, optional): number of learnable a parameters. Default: ``1``
        b (float, optional): fixed bound parameter. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PLU.png

    Examples::

        >>> m = torch_activation.PLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, num_parameters: int = 1, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.ones(num_parameters) * 0.1)
        self.b = b

    def _forward(self, x) -> Tensor:
        b = self.b
        return torch.max(self.a * (x + b) - b, torch.min(self.a * (x - b) + b, x))


@register_activation
class VAF(BaseActivation):
    r"""
    Applies the Variable Activation Function (VAF):

    :math:`\text{VAF}(z_l) = \sum_{j=1}^{J} a_{l,j} \cdot g(b_{l,j} z_l + c_{l,j}) + a_{l,0}`

    where :math:`g` is tanh and all coefficients are learnable.

    Args:
        J (int, optional): number of basis functions. Default: ``3``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/VAF.png

    Examples::

        >>> m = torch_activation.VAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, J: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.J = J
        a_init = torch.zeros(J + 1)
        a_init[1] = 1.0
        self.a = nn.Parameter(a_init)
        self.b = nn.Parameter(torch.ones(J))
        self.c = nn.Parameter(torch.zeros(J))

    def _forward(self, x) -> Tensor:
        out = self.a[0].expand_as(x)
        for j in range(self.J):
            out = out + self.a[j + 1] * torch.tanh(self.b[j] * x + self.c[j])
        return out


@register_activation
class FAB(BaseActivation):
    r"""
    Applies the Flexible Activation Bag (FAB) — a weighted sum of basis activations:

    :math:`\text{FAB}(z) = \sum_{k} w_k \cdot f_k(z)`

    where :math:`f_k` are fixed basis activations (relu, tanh, sigmoid, identity) and :math:`w_k` are learnable weights.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/FAB.png

    Examples::

        >>> m = torch_activation.FAB()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.ones(4) / 4)

    def _forward(self, x) -> Tensor:
        w = torch.softmax(self.w, dim=0)
        return w[0] * F.relu(x) + w[1] * torch.tanh(x) + w[2] * torch.sigmoid(x) + w[3] * x


@register_activation
class KAF(BaseActivation):
    r"""
    Applies the Kernel Activation Function (KAF):

    :math:`\text{KAF}(z_i) = \sum_{j=1}^{D} a_{i,j} \exp\!\left(-\gamma (z_i - d_j)^2\right)`

    where :math:`d_j` is a fixed grid, :math:`\gamma` is a fixed bandwidth, and :math:`a_{i,j}` are learnable mixing coefficients.

    Args:
        D (int, optional): number of dictionary elements. Default: ``20``
        gamma (float, optional): Gaussian bandwidth. Default: ``1.0``
        bound (float, optional): grid spans [-bound, bound]. Default: ``3.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/KAF.png

    Examples::

        >>> m = torch_activation.KAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, D: int = 20, gamma: float = 1.0, bound: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.gamma = gamma
        self.bound = bound
        d = torch.linspace(-bound, bound, D)
        self.register_buffer("d", d)
        self.a = nn.Parameter(d / D)

    def extra_repr(self):
        return f"D={self.D}, gamma={self.gamma:.4f}, bound={self.bound:.4f}"

    def _forward(self, x) -> Tensor:
        diff = x.unsqueeze(-1) - self.d
        k = torch.exp(-self.gamma * diff**2)
        return (k * self.a).sum(-1)


@register_activation
class RTPReLU(BaseActivation):
    r"""
    Applies the Random Threshold PReLU (RTPReLU):

    :math:`\text{RTPReLU}(z_i) = z_i \text{ if } z_i + b_i \geq 0,\; \text{else } z_i / a`

    where :math:`b_i \sim \mathcal{N}(0, \sigma^2)` during training, and the threshold is zero at eval time.

    Args:
        a (float, optional): learnable slope for negative part. Default: ``4.0``
        sigma (float, optional): std of the threshold noise during training. Default: ``0.5``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RTPReLU.png

    Examples::

        >>> m = torch_activation.RTPReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 4.0, sigma: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.a = nn.Parameter(torch.tensor(a))
        self.sigma = sigma

    def _forward(self, x) -> Tensor:
        if self.training:
            b = torch.randn_like(x) * self.sigma
            mask = (x + b) >= 0
        else:
            mask = x >= 0
        a = self.a.abs().clamp(min=1e-6)
        return torch.where(mask, x, x / a)


@register_activation
class DYReLU(BaseActivation):
    r"""
    Applies the Dynamic ReLU (DYReLU):

    :math:`\text{DYReLU}(z_i) = \max_{1 \leq k \leq K}\!\left(a_{i,k} z_i + b_{i,k}\right)`

    where the K linear functions are parameterized by learnable coefficients.

    Args:
        K (int, optional): number of linear pieces. Default: ``2``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DYReLU.png

    Examples::

        >>> m = torch_activation.DYReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, K: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        a_init = torch.zeros(K)
        a_init[0] = 1.0
        self.a = nn.Parameter(a_init)
        self.b = nn.Parameter(torch.zeros(K))

    def _forward(self, x) -> Tensor:
        pieces = [self.a[k] * x + self.b[k] for k in range(self.K)]
        return torch.stack(pieces, dim=-1).max(dim=-1).values


@register_activation
class FunPReLU(BaseActivation):
    r"""
    Applies the Functional Parametric ReLU (FunPReLU):

    :math:`\text{FunPReLU}(z) = \max(z, t(z))`

    where :math:`t(z) = w z + c` is a learnable linear threshold function.

    Args:
        init_w (float, optional): initial slope for t. Default: ``0.25``
        init_c (float, optional): initial bias for t. Default: ``0.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/FunPReLU.png

    Examples::

        >>> m = torch_activation.FunPReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, init_w: float = 0.25, init_c: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.tensor(init_w))
        self.c = nn.Parameter(torch.tensor(init_c))

    def _forward(self, x) -> Tensor:
        return torch.max(x, self.w * x + self.c)


@register_activation
class PLAF(BaseActivation):
    r"""
    Applies the Piecewise Linear Approximation Function (PLAF):

    .. math::

        \text{PLAF}(z) = \begin{cases}
        z - (1 - 1/d), & z \geq 1 \\
        (1/d)|z|^d \cdot \text{sign}(z), & -1 \leq z < 1 \\
        z + (1 - 1/d), & z < -1
        \end{cases}

    Args:
        d (float, optional): shaping parameter. Default: ``2.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PLAF.png

    Examples::

        >>> m = torch_activation.PLAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, d: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.d = d

    def _forward(self, x) -> Tensor:
        d = self.d
        offset = 1 - 1 / d
        mid = (1 / d) * x.abs() ** d * x.sign()
        return torch.where(x >= 1, x - offset, torch.where(x < -1, x + offset, mid))


@register_activation
class HybridChaoticAF(BaseActivation):
    r"""
    Applies the Hybrid Chaotic Activation Function (HybridChaoticAF):

    :math:`\text{HybridChaoticAF}(z) = \tanh(z) + a \sin(\pi z)`

    Args:
        a (float, optional): amplitude of chaotic perturbation. Default: ``0.1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HybridChaoticAF.png

    Examples::

        >>> m = torch_activation.HybridChaoticAF()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x) -> Tensor:
        return torch.tanh(x) + self.a * torch.sin(math.pi * x)
