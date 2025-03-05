import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable

from torch_activation import register_activation

@register_activation
class ShiLU(nn.Module):
    r"""
    Applies the ShiLU activation function:

    :math:`\text{ShiLU}(x) = \alpha \cdot \text{ReLU}(x) + \beta`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        alpha (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        beta (float, optional): Bias term added to the output. Default: 0.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ShiLU.png

    Examples::

        >>> m = torch_activation.ShiLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ShiLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, inplace: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(Tensor([alpha]))
        self.beta = nn.Parameter(Tensor([beta]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            F.relu_(x)
            x.mul_(self.alpha)
            x.add_(self.beta)
            return x
        else:
            return self.alpha * F.relu(x) + self.beta


@register_activation
class StarReLU(nn.Module):
    r"""
    Applies the element-wise function:

    :math:`\text{StarReLU}(x) = s \cdot \text{ReLU}(x)^2 + b`

     See: https://doi.org/10.48550/arXiv.2210.13452

    Args:
        s (float, optional): Scaled factor for StarReLU, shared across channel. Default: 0.8944
        b (float, optional): Bias term for StarReLU, shared across channel. Default: -0.4472
        learnable (bool, optional): optionally make ``s`` and ``b`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/StarReLU.png

    Examples::

        >>> m = torch_activation.StarReLU(s=1.0, b=0.0)
        >>> x = torch.randn(3, 384, 384)
        >>> output = m(x)

        >>> m = torch_activation.StarReLU(learnable=True, inplace=True)
        >>> x = torch.randn(3, 384, 384)
        >>> m(x)
    """

    def __init__(
        self,
        s: float = 0.8944,
        b: float = -0.4472,
        learnable: bool = False,
        inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.s = nn.Parameter(Tensor([s]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.s = Tensor([s])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            return F.relu_(x).pow_(2).mul_(self.s).add_(self.b)
        else:
            return self.s * F.relu(x).pow(2) + self.b


@register_activation
class DELU(nn.Module):
    r"""
    Applies the DELU activation function:

    :math:`\text{DELU}(x) = \begin{cases} \text{SiLU}(x), x \leqslant 0 \\x(n-1), \text{otherwise} \end{cases}`


     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        n (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DELU.png

    Examples:
        >>> m = nn.DELU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.DELU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, n: float = 1.0, inplace: bool = False):
        super(DELU, self).__init__()
        self.n = torch.nn.Parameter(Tensor([n]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        return torch.where(
            x <= 0, F.silu(x), (self.n + 0.5) * x + torch.abs(torch.exp(-x) - 1)
        )

    def _forward_inplace(self, x):
        x[x <= 0] = F.silu(x[x <= 0])
        x[x > 0] = (self.n + 0.5) * x[x > 0] + torch.abs(torch.exp(-x[x > 0]) - 1)
        return x


@register_activation
class PReLU(nn.Module):
    r"""
    Applies the Parametric Rectified Linear Unit function:

    :math:`\text{PReLU}(x) = \begin{cases} x, & x \geq 0 \\ \frac{x}{a}, & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor for the negative part of the input. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PReLU(a=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, learnable: bool = False, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = x[mask] / self.a
            return x
        else:
            return torch.where(x >= 0, x, x / self.a)


@register_activation
class PReLUPlus(nn.Module):
    r"""
    Applies the Positive Parametric Rectified Linear Unit function:

    :math:`\text{PReLU+}(x) = \begin{cases} a \cdot x, & x \geq 0 \\ 0, & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor for the positive part of the input. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PReLUPlus(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PReLUPlus(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, learnable: bool = False, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x.mul_(self.a)
            x[mask] = 0
            return x
        else:
            return torch.where(x >= 0, self.a * x, torch.zeros_like(x))


@register_activation
class MarReLU(nn.Module):
    r"""
    Applies the Margin ReLU activation function:

    :math:`\text{MarReLU}(x) = \max(x, a) = \begin{cases} x, & x - a \geq 0 \\ a, & x - a < 0 \end{cases}`

    Args:
        a (float, optional): Margin threshold. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.MarReLU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MarReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, a: float = 0.0, learnable: bool = False, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < self.a
            x[mask] = self.a
            return x
        else:
            return torch.maximum(x, self.a * torch.ones_like(x))


@register_activation
class RPReLU(nn.Module):
    r"""
    Applies the React-PReLU activation function:

    :math:`\text{RPReLU}(x) = \begin{cases} x - a + b, & x \geq a \\ c(x - a) + b, & x < a \end{cases}`

    Args:
        a (float, optional): Threshold parameter. Default: 0.0
        b (float, optional): Bias parameter. Default: 0.0
        c (float, optional): Scaling factor for the negative part. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.RPReLU(a=0.5, b=0.1, c=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RPReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.0, 
        b: float = 0.0, 
        c: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < self.a
            x_minus_a = x - self.a
            x[~mask] = x_minus_a[~mask] + self.b
            x[mask] = self.c * x_minus_a[mask] + self.b
            return x
        else:
            x_minus_a = x - self.a
            return torch.where(x >= self.a, x_minus_a + self.b, self.c * x_minus_a + self.b)


@register_activation
class LeLeLU(nn.Module):
    r"""
    Applies the Leaky Learnable ReLU activation function:

    :math:`\text{LeLeLU}(x) = \begin{cases} a \cdot x, & x \geq 0 \\ 0.01 \cdot a \cdot x, & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor. Default: 1.0
        negative_slope (float, optional): Controls the slope of the negative part. Default: 0.01
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.LeLeLU(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LeLeLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        negative_slope: float = 0.01, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.negative_slope = negative_slope
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x.mul_(self.a)
            x[mask].mul_(self.negative_slope)
            return x
        else:
            return torch.where(x >= 0, self.a * x, self.negative_slope * self.a * x)


@register_activation
class PREU(nn.Module):
    r"""
    Applies the Parametric Rectified Exponential Unit activation function:

    :math:`\text{PREU}(x) = \begin{cases} a \cdot x, & x \geq 0 \\ a \cdot x \cdot \exp(b \cdot x), & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor. Default: 1.0
        b (float, optional): Exponential factor for negative values. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PREU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PREU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x.mul_(self.a)
            x[mask].mul_(torch.exp(self.b * x[mask]))
            return x
        else:
            return torch.where(x >= 0, self.a * x, self.a * x * torch.exp(self.b * x))


@register_activation
class RTReLU(nn.Module):
    r"""
    Applies the Randomly Translational PReLU activation function:

    :math:`\text{RT-PReLU}(x) = \begin{cases} x, & x + b \geq 0 \\ \frac{x}{a}, & x + b < 0 \end{cases}`

    where :math:`b \sim N(0, \sigma^2)`

    Args:
        a (float, optional): Scaling factor for the negative part. Default: 1.0
        sigma (float, optional): Standard deviation for random translation. Default: 0.75
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.RTReLU(a=0.1, sigma=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RTReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        sigma: float = 0.75, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.sigma = sigma
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        b = torch.randn_like(x) * self.sigma
        if self.inplace:
            mask = x + b < 0
            x[mask] = x[mask] / self.a
            return x
        else:
            return torch.where(x + b >= 0, x, x / self.a)


@register_activation
class SMU(nn.Module):
    r"""
    Applies the Smooth Maximum Unit activation function:

    :math:`\text{SMU}(x) = \frac{(1 + a)x + (1 - a)x \cdot \text{erf}(b (1 - a)x)}{2}`

    Args:
        a (float, optional): Shape parameter. Default: 0.25
        b (float, optional): Smoothing parameter. Default: 25.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SMU(a=0.25, b=25.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SMU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        a: float = 0.25, 
        b: float = 25.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        term1 = (1 + self.a) * x
        term2 = (1 - self.a) * x * torch.erf(self.b * (1 - self.a) * x)
        
        if self.inplace:
            x.copy_((term1 + term2) / 2)
            return x
        else:
            return (term1 + term2) / 2


@register_activation
class SAU(nn.Module):
    r"""
    Applies the Smooth Activation Unit function:

    :math:`\text{SAU}(x) = (\text{PReLU}_{a} * \phi_{b})(x) = \frac{1}{2b \sqrt{\pi}} \exp\left(-\frac{b^2 x^2}{2}\right) + \frac{1}{2}\left(1 - \frac{a}{x} + \frac{x \cdot \text{erf}(b x / \sqrt{2})}{2}\right)`

    Args:
        a (float, optional): PReLU parameter. Default: 1.0
        b (float, optional): Smoothing parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SAU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SAU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        learnable: bool = False
    ):
        super().__init__()
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Handle potential division by zero
        safe_x = torch.where(x == 0, torch.ones_like(x) * 1e-10, x)
        
        term1 = 1 / (2 * self.b * torch.sqrt(torch.tensor(torch.pi))) * torch.exp(-self.b**2 * x**2 / 2)
        term2 = 0.5 * (1 - self.a / safe_x + x * torch.erf(self.b * x / torch.sqrt(torch.tensor(2.0))))
        
        return term1 + term2


@register_activation
class ProbAct(nn.Module):
    r"""
    Applies the Probabilistic Activation function:

    :math:`\text{ProbAct}(x) = g(x) + \sigma \cdot e`

    where :math:`e \sim N(0, 1)` and :math:`g(x)` is a base activation function.

    Args:
        base_activation (callable, optional): Base activation function. Default: ``torch.nn.functional.relu``
        sigma (float, optional): Standard deviation of the noise. Default: 0.1
        learnable (bool, optional): optionally make ``sigma`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ProbAct(sigma=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ProbAct(base_activation=torch.sigmoid, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        base_activation: Callable = F.relu, 
        sigma: float = 0.1, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.base_activation = base_activation
        self.inplace = inplace
        if learnable:
            self.sigma = nn.Parameter(Tensor([sigma]))
        else:
            self.sigma = Tensor([sigma])

    def forward(self, x) -> Tensor:
        activated = self.base_activation(x)
        noise = torch.randn_like(x) * self.sigma
        
        if self.inplace and hasattr(activated, 'add_'):
            activated.add_(noise)
            return activated
        else:
            return activated + noise


@register_activation
class ReLUProbAct(nn.Module):
    r"""
    Applies the ReLU-based Probabilistic Activation function:

    :math:`\text{ReLUProbAct}(x) = \max(0, x) + \sigma \cdot e`

    where :math:`e \sim N(0, 1)`

    Args:
        sigma (float, optional): Standard deviation of the noise. Default: 0.1
        learnable (bool, optional): optionally make ``sigma`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ReLUProbAct(sigma=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReLUProbAct(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        sigma: float = 0.1, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.sigma = nn.Parameter(Tensor([sigma]))
        else:
            self.sigma = Tensor([sigma])

    def forward(self, x) -> Tensor:
        if self.inplace:
            x.clamp_(min=0)
            x.add_(torch.randn_like(x) * self.sigma)
            return x
        else:
            return F.relu(x) + torch.randn_like(x) * self.sigma


@register_activation
class AOAF(nn.Module):
    r"""
    Applies the Adaptive Offset Activation Function:

    :math:`\text{AOAF}(x) = \max(0, x - b \cdot a) + c \cdot a`

    Args:
        a (float, optional): Adaptive parameter. Default: 0.1
        b (float, optional): Offset scaling parameter. Default: 0.17
        c (float, optional): Bias scaling parameter. Default: 0.17
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.AOAF(a=0.1, b=0.17, c=0.17)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AOAF(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.1, 
        b: float = 0.17, 
        c: float = 0.17, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.b = b
        self.c = c
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        offset = self.b * self.a
        bias = self.c * self.a
        
        if self.inplace:
            x.sub_(offset).clamp_(min=0).add_(bias)
            return x
        else:
            return torch.clamp(x - offset, min=0) + bias


@register_activation
class DLReLU(nn.Module):
    r"""
    Applies the Dynamic Leaky ReLU function:

    :math:`\text{DLReLU}(x) = \begin{cases} x, & x \geq 0 \\ a \cdot b_t \cdot x, & x < 0 \end{cases}`

    where :math:`b_t = \text{MSE}_{t-1}` is the mean squared error from the previous iteration.

    Args:
        a (float, optional): Scaling factor. Default: 0.01
        mse (float, optional): Initial MSE value. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.DLReLU(a=0.02, mse=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DLReLU(learnable=True, inplace=True)
        >>> m.update_mse(0.3)  # Update MSE value
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.01, 
        mse: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.mse = mse
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def update_mse(self, mse: float):
        """Update the MSE value for the dynamic slope."""
        self.mse = mse

    def forward(self, x) -> Tensor:
        negative_slope = self.a * self.mse
        
        if self.inplace:
            mask = x < 0
            x[mask].mul_(negative_slope)
            return x
        else:
            return torch.where(x >= 0, x, negative_slope * x)


@register_activation
class ExpDLReLU(nn.Module):
    r"""
    Applies the Exponential Dynamic Leaky ReLU function:

    :math:`\text{exp-DLReLU}(x) = \begin{cases} x, & x \geq 0 \\ a \cdot c_t \cdot x, & x < 0 \end{cases}`

    where :math:`c_t = \exp(-\text{MSE}_{t-1})` is the exponential of the negative mean squared error from the previous iteration.

    Args:
        a (float, optional): Scaling factor. Default: 0.01
        mse (float, optional): Initial MSE value. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ExpDLReLU(a=0.02, mse=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ExpDLReLU(learnable=True, inplace=True)
        >>> m.update_mse(0.3)  # Update MSE value
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.01, 
        mse: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.mse = mse
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def update_mse(self, mse: float):
        """Update the MSE value for the dynamic slope."""
        self.mse = mse

    def forward(self, x) -> Tensor:
        negative_slope = self.a * torch.exp(-torch.tensor(self.mse))
        
        if self.inplace:
            mask = x < 0
            x[mask].mul_(negative_slope)
            return x
        else:
            return torch.where(x >= 0, x, negative_slope * x)


@register_activation
class DReLU(nn.Module):
    r"""
    Applies the Dynamic ReLU function:

    :math:`\text{DReLU}(x) = \begin{cases} x, & x - a \geq 0 \\ a, & x - a < 0 \end{cases}`

    Args:
        a (float, optional): Threshold parameter. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.DReLU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < self.a
            x[mask] = self.a
            return x
        else:
            return torch.where(x - self.a >= 0, x, self.a)


@register_activation
class FReLU(nn.Module):
    r"""
    Applies the Flexible ReLU function:

    :math:`\text{FReLU}(x) = \text{ReLU}(x) + b = \begin{cases} x + b, & x \geq 0 \\ b, & x < 0 \end{cases}`

    Args:
        b (float, optional): Bias parameter. Default: 0.0
        learnable (bool, optional): optionally make ``b`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.FReLU(b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.FReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        b: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = 0
            x.add_(self.b)
            return x
        else:
            return F.relu(x) + self.b


@register_activation
class AdaptiveHardTanh(nn.Module):
    r"""
    Applies the Adaptive HardTanh function:

    :math:`\text{AdaptiveHardTanh}(x) = \text{HardTanh}(a_t \cdot (x - b))`

    Args:
        a (float, optional): Scaling parameter. Default: 1.0
        b (float, optional): Shift parameter. Default: 0.0
        min_val (float, optional): Minimum value of the HardTanh. Default: -1.0
        max_val (float, optional): Maximum value of the HardTanh. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.AdaptiveHardTanh(a=2.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AdaptiveHardTanh(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.0, 
        min_val: float = -1.0,
        max_val: float = 1.0,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.min_val = min_val
        self.max_val = max_val
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        scaled_shifted = self.a * (x - self.b)
        
        if self.inplace:
            scaled_shifted.clamp_(min=self.min_val, max=self.max_val)
            return scaled_shifted
        else:
            return torch.clamp(scaled_shifted, min=self.min_val, max=self.max_val)


@register_activation
class AReLU(nn.Module):
    r"""
    Applies the Attention-based ReLU function:

    :math:`\text{AReLU}(x) = \begin{cases} (1 + \sigma(b)) \cdot x, & x \geq 0 \\ C(a) \cdot x, & x < 0 \end{cases}`

    where :math:`\sigma` is the sigmoid function and :math:`C(a)` is a function of parameter :math:`a`.

    Args:
        a (float, optional): Parameter for negative slope. Default: 0.9
        b (float, optional): Parameter for positive slope. Default: 2.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.AReLU(a=0.9, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.9, 
        b: float = 2.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        positive_scale = 1 + torch.sigmoid(self.b)
        negative_scale = self.a  # C(a) = a in the simplest case
        
        if self.inplace:
            mask = x < 0
            x[~mask].mul_(positive_scale)
            x[mask].mul_(negative_scale)
            return x
        else:
            return torch.where(x >= 0, positive_scale * x, negative_scale * x)


@register_activation
class DPReLU(nn.Module):
    r"""
    Applies the Dual Parametric ReLU function:

    :math:`\text{DPReLU}(x) = \begin{cases} a \cdot x, & x \geq 0 \\ b \cdot x, & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor for positive values. Default: 1.0
        b (float, optional): Scaling factor for negative values. Default: 0.01
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.DPReLU(a=1.0, b=0.01)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DPReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.01, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask].mul_(self.a)
            x[mask].mul_(self.b)
            return x
        else:
            return torch.where(x >= 0, self.a * x, self.b * x)


@register_activation
class DualLine(nn.Module):
    r"""
    Applies the Dual Line activation function:

    :math:`\text{DualLine}(x) = \begin{cases} a \cdot x + m, & x \geq 0 \\ b \cdot x + m, & x < 0 \end{cases}`

    Args:
        a (float, optional): Slope for positive values. Default: 1.0
        b (float, optional): Slope for negative values. Default: 0.01
        m (float, optional): Bias term. Default: -0.22
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.DualLine(a=1.0, b=0.01, m=-0.22)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DualLine(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.01, 
        m: float = -0.22, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.m = nn.Parameter(Tensor([m]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.m = Tensor([m])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask].mul_(self.a).add_(self.m)
            x[mask].mul_(self.b).add_(self.m)
            return x
        else:
            return torch.where(x >= 0, self.a * x + self.m, self.b * x + self.m)


@register_activation
class PiLU(nn.Module):
    r"""
    Applies the Piecewise Linear Unit function:

    :math:`\text{PiLU}(x) = \begin{cases} a \cdot x + c(1 - a), & x \geq c \\ b \cdot x + c(1 - b), & x < c \end{cases}`

    Args:
        a (float, optional): Slope for values above threshold. Default: 1.0
        b (float, optional): Slope for values below threshold. Default: 0.01
        c (float, optional): Threshold parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PiLU(a=1.0, b=0.01, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PiLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.01, 
        c: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        bias_a = self.c * (1 - self.a)
        bias_b = self.c * (1 - self.b)
        
        if self.inplace:
            mask = x < self.c
            x[~mask].mul_(self.a).add_(bias_a)
            x[mask].mul_(self.b).add_(bias_b)
            return x
        else:
            return torch.where(x >= self.c, self.a * x + bias_a, self.b * x + bias_b)


@register_activation
class DPAF(nn.Module):
    r"""
    Applies the Dual Parametric Activation Function:

    :math:`\text{DPAF}(x) = \begin{cases} a \cdot g(x) + m, & x \geq 0 \\ g(x) + m, & x < 0 \end{cases}`

    where :math:`g(x)` is a base activation function.

    Args:
        a (float, optional): Scaling factor for positive values. Default: 1.0
        m (float, optional): Bias term. Default: 0.0
        base_activation (callable, optional): Base activation function. Default: ``torch.nn.functional.relu``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.DPAF(a=1.5, m=0.1, base_activation=torch.tanh)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DPAF(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        m: float = 0.0, 
        base_activation: Callable = F.relu,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.base_activation = base_activation
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.m = nn.Parameter(Tensor([m]))
        else:
            self.a = Tensor([a])
            self.m = Tensor([m])

    def forward(self, x) -> Tensor:
        activated = self.base_activation(x)
        
        if self.inplace and hasattr(activated, 'add_'):
            mask = x >= 0
            activated[mask].mul_(self.a)
            activated.add_(self.m)
            return activated
        else:
            return torch.where(x >= 0, self.a * activated + self.m, activated + self.m)


@register_activation
class FPAF(nn.Module):
    r"""
    Applies the Fully Parameterized Activation Function:

    :math:`\text{FPAF}(x) = \begin{cases} a \cdot g_1(x), & x \geq 0 \\ b \cdot g_2(x), & x < 0 \end{cases}`

    where :math:`g_1(x)` and :math:`g_2(x)` are base activation functions.

    Args:
        a (float, optional): Scaling factor for positive values. Default: 1.0
        b (float, optional): Scaling factor for negative values. Default: 1.0
        pos_activation (callable, optional): Activation for positive values. Default: ``torch.nn.functional.relu``
        neg_activation (callable, optional): Activation for negative values. Default: ``torch.nn.functional.relu``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.FPAF(a=1.0, b=0.5, pos_activation=torch.tanh, neg_activation=torch.sigmoid)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.FPAF(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        pos_activation: Callable = F.relu,
        neg_activation: Callable = F.relu,
        learnable: bool = False
    ):
        super().__init__()
        self.pos_activation = pos_activation
        self.neg_activation = neg_activation
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = torch.zeros_like(x)
        if pos_mask.any():
            result[pos_mask] = self.a * self.pos_activation(x[pos_mask])
        if neg_mask.any():
            result[neg_mask] = self.b * self.neg_activation(x[neg_mask])
        
        return result


@register_activation
class EPReLU(nn.Module):
    r"""
    Applies the Elastic PReLU function:

    :math:`\text{EPReLU}(x) = \begin{cases} k \cdot x, & x \geq 0 \\ \frac{x}{a}, & x < 0 \end{cases}`

    where :math:`k \sim U(1 - \alpha, 1 + \alpha)` is sampled from a uniform distribution.

    Args:
        a (float, optional): Scaling factor for negative values. Default: 1.0
        alpha (float, optional): Range parameter for the uniform distribution. Default: 0.1
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.EPReLU(a=0.5, alpha=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.EPReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        alpha: float = 0.1, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.alpha = alpha
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # Sample k from uniform distribution
        k = 1 + (2 * torch.rand_like(x) - 1) * self.alpha
        
        if self.inplace:
            mask = x < 0
            x[~mask].mul_(k[~mask])
            x[mask].div_(self.a)
            return x
        else:
            return torch.where(x >= 0, k * x, x / self.a)


@register_activation
class PairedReLU(nn.Module):
    r"""
    Applies the Paired ReLU function:

    :math:`\text{PairedReLU}(x) = \begin{pmatrix} \max(a \cdot x - b, 0) \\ \max(c \cdot x - d, 0) \end{pmatrix}`

    Args:
        a (float, optional): Scaling factor for first component. Default: 0.5
        b (float, optional): Bias for first component. Default: 0.0
        c (float, optional): Scaling factor for second component. Default: -0.5
        d (float, optional): Bias for second component. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``

    Shape:
        - Input: :math:`(N, C, *)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(N, 2*C, *)`, doubling the channel dimension.

    Examples::

        >>> m = torch_activation.PairedReLU(a=0.5, c=-0.5)
        >>> x = torch.randn(2, 3, 4, 5)
        >>> output = m(x)  # shape: [2, 6, 4, 5]

        >>> m = torch_activation.PairedReLU(learnable=True)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)  # shape: [2, 6]
    """

    def __init__(
        self, 
        a: float = 0.5, 
        b: float = 0.0, 
        c: float = -0.5, 
        d: float = 0.0, 
        learnable: bool = False
    ):
        super().__init__()
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
            self.d = nn.Parameter(Tensor([d]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])
            self.d = Tensor([d])

    def forward(self, x) -> Tensor:
        # First component: max(a*x - b, 0)
        y1 = F.relu(self.a * x - self.b)
        
        # Second component: max(c*x - d, 0)
        y2 = F.relu(self.c * x - self.d)
        
        # Stack the outputs along the channel dimension
        if x.dim() <= 1:
            return torch.cat([y1, y2], dim=0)
        else:
            return torch.cat([y1, y2], dim=1)


@register_activation
class Tent(nn.Module):
    r"""
    Applies the Tent activation function:

    :math:`\text{Tent}(x) = \max(0, a - |x|)`

    Args:
        a (float, optional): Width parameter of the tent. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.Tent(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Tent(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            x.abs_().neg_().add_(self.a).clamp_(min=0)
            return x
        else:
            return torch.clamp(self.a - torch.abs(x), min=0)


@register_activation
class Hat(nn.Module):
    r"""
    Applies the Hat activation function:

    :math:`\text{Hat}(x) = \begin{cases} 
        0, & x < 0 \\
        x, & 0 \leq x \leq \frac{a}{2} \\
        a - x, & \frac{a}{2} \leq x \leq a \\
        0, & x > a 
    \end{cases}`

    Args:
        a (float, optional): Width parameter of the hat. Default: 2.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.Hat(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Hat(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 2.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        half_a = self.a / 2
        
        if self.inplace:
            # Create a copy to avoid modifying the original during computation
            result = x.clone()
            
            # Set values for different regions
            result[x < 0] = 0
            mask_middle = (x >= half_a) & (x <= self.a)
            result[mask_middle] = self.a - x[mask_middle]
            result[x > self.a] = 0
            
            # Copy back to original tensor if needed
            if self.inplace:
                x.copy_(result)
                return x
            return result
        else:
            # Create masks for different regions
            mask_lower = (x >= 0) & (x < half_a)
            mask_middle = (x >= half_a) & (x <= self.a)
            
            # Initialize with zeros
            result = torch.zeros_like(x)
            
            # Set values for different regions
            result[mask_lower] = x[mask_lower]
            result[mask_middle] = self.a - x[mask_middle]
            
            return result


@register_activation
class RMAF(nn.Module):
    r"""
    Applies the ReLU Memristor-like Activation Function:

    :math:`\text{RMAF}(x) = b \left(\frac{1}{0.25(1 + \exp(-x)) + 0.75}\right)^c a \cdot x`

    Args:
        a (float, optional): Scaling parameter. Default: 1.0
        b (float, optional): Scaling parameter. Default: 1.0
        c (float, optional): Exponent parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.RMAF(a=1.0, b=1.0, c=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RMAF(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        # Calculate the memristor-like term
        memristor_term = (1 / (0.25 * (1 + torch.exp(-x)) + 0.75)) ** self.c
        
        # Apply scaling
        result = self.b * memristor_term * self.a * x
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PTELU(nn.Module):
    r"""
    Applies the Parametric Tanh Exponential Linear Unit function:

    :math:`\text{PTELU}(x) = \begin{cases} x, & x \geq 0 \\ a \cdot \tanh(b \cdot x), & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor for negative values. Default: 1.0
        b (float, optional): Scaling factor inside tanh for negative values. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PTELU(a=0.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PTELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = self.a * torch.tanh(self.b * x[mask])
            return x
        else:
            return torch.where(x >= 0, x, self.a * torch.tanh(self.b * x))


@register_activation
class TaLU(nn.Module):
    r"""
    Applies the Tangent Linear Unit function:

    :math:`\text{TaLU}(x) = \begin{cases} 
        x, & x \geq 0 \\
        \tanh(x), & a < x < 0 \\
        \tanh(a), & x \leq a 
    \end{cases}`

    Args:
        a (float, optional): Lower threshold parameter. Default: -1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TaLU(a=-1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TaLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = -1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        tanh_a = torch.tanh(self.a)
        
        if self.inplace:
            mask_middle = (x < 0) & (x > self.a)
            mask_lower = x <= self.a
            
            x[mask_middle] = torch.tanh(x[mask_middle])
            x[mask_lower] = tanh_a
            
            return x
        else:
            # Create masks for different regions
            mask_upper = x >= 0
            mask_middle = (x < 0) & (x > self.a)
            mask_lower = x <= self.a
            
            # Initialize result tensor
            result = torch.zeros_like(x)
            
            # Apply different functions to different regions
            result[mask_upper] = x[mask_upper]
            result[mask_middle] = torch.tanh(x[mask_middle])
            result[mask_lower] = tanh_a
            
            return result


@register_activation
class PTaLU(nn.Module):
    r"""
    Applies the Parametric Tangent Linear Unit function:

    :math:`\text{PTaLU}(x) = \begin{cases} 
        x, & x \geq b \\
        \tanh(x), & a < x < b \\
        \tanh(a), & x \leq a 
    \end{cases}`

    Args:
        a (float, optional): Lower threshold parameter. Default: -0.75
        b (float, optional): Upper threshold parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PTaLU(a=-0.75, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PTaLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = -0.75, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        tanh_a = torch.tanh(self.a)
        
        if self.inplace:
            mask_middle = (x < self.b) & (x > self.a)
            mask_lower = x <= self.a
            
            x[mask_middle] = torch.tanh(x[mask_middle])
            x[mask_lower] = tanh_a
            
            return x
        else:
            # Create masks for different regions
            mask_upper = x >= self.b
            mask_middle = (x < self.b) & (x > self.a)
            mask_lower = x <= self.a
            
            # Initialize result tensor
            result = torch.zeros_like(x)
            
            # Apply different functions to different regions
            result[mask_upper] = x[mask_upper]
            result[mask_middle] = torch.tanh(x[mask_middle])
            result[mask_lower] = tanh_a
            
            return result


@register_activation
class TanhLU(nn.Module):
    r"""
    Applies the TanhLU activation function:

    :math:`\text{TanhLU}(x) = a \cdot \tanh(b \cdot x) + c \cdot x`

    Args:
        a (float, optional): Scaling factor for tanh component. Default: 1.0
        b (float, optional): Scaling factor inside tanh. Default: 1.0
        c (float, optional): Scaling factor for linear component. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TanhLU(a=0.5, b=2.0, c=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TanhLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        result = self.a * torch.tanh(self.b * x) + self.c * x
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TeLU(nn.Module):
    r"""
    Applies the Tanh Exponential Linear Unit function:

    :math:`\text{TeLU}(x) = x \cdot \tanh(\text{ELU}(a \cdot x))`

    where ELU is the Exponential Linear Unit.

    Args:
        a (float, optional): Scaling factor inside ELU. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TeLU(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TeLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # Apply ELU
        elu_output = F.elu(self.a * x)
        
        # Apply tanh and multiply by x
        result = x * torch.tanh(elu_output)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TReLU(nn.Module):
    r"""
    Applies the Tanh-based ReLU function:

    :math:`\text{TReLU}(x) = \begin{cases} x, & x \geq 0 \\ \tanh(b \cdot x), & x < 0 \end{cases}`

    Args:
        b (float, optional): Scaling factor inside tanh for negative values. Default: 1.0
        learnable (bool, optional): optionally make ``b`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TReLU(b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = torch.tanh(self.b * x[mask])
            return x
        else:
            return torch.where(x >= 0, x, torch.tanh(self.b * x))


@register_activation
class TReLU2(nn.Module):
    r"""
    Applies the Tanh-based ReLU variant 2 function:

    :math:`\text{TReLU2}(x) = \begin{cases} x, & x \geq 0 \\ a \cdot \tanh(x), & x < 0 \end{cases}`

    Args:
        a (float, optional): Scaling factor for tanh component. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TReLU2(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TReLU2(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = self.a * torch.tanh(x[mask])
            return x
        else:
            return torch.where(x >= 0, x, self.a * torch.tanh(x))

@register_activation
class ReLTanh(nn.Module):
    r"""
    Applies the Rectified Linear Tanh function:

    :math:`\text{ReLTanh}(x) = \begin{cases} 
        \tanh'(a)(x - a) + \tanh(a), & x \leq a \\
        \tanh(x), & a < x < b \\
        \tanh'(b)(x - b) + \tanh(b), & x \geq b 
    \end{cases}`

    where :math:`\tanh'(x) = \frac{4}{(\exp(x) + \exp(-x))^2}` is the derivative of tanh.

    Args:
        a (float, optional): Lower threshold parameter. Default: -1.5
        b (float, optional): Upper threshold parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ReLTanh(a=-1.5, b=0.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReLTanh(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = -1.5, 
        b: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _tanh_derivative(self, x):
        # Derivative of tanh(x) = 1 - tanh^2(x) = 4 / (e^x + e^-x)^2
        return 4 / (torch.exp(x) + torch.exp(-x))**2

    def forward(self, x) -> Tensor:
        tanh_a = torch.tanh(self.a)
        tanh_b = torch.tanh(self.b)
        tanh_deriv_a = self._tanh_derivative(self.a)
        tanh_deriv_b = self._tanh_derivative(self.b)
        
        # Linear approximation at a
        lower_linear = tanh_deriv_a * (x - self.a) + tanh_a
        
        # Linear approximation at b
        upper_linear = tanh_deriv_b * (x - self.b) + tanh_b
        
        if self.inplace:
            # Create a copy to avoid modifying during computation
            result = x.clone()
            
            # Apply different functions to different regions
            mask_lower = x <= self.a
            mask_middle = (x > self.a) & (x < self.b)
            mask_upper = x >= self.b
            
            result[mask_lower] = lower_linear[mask_lower]
            result[mask_middle] = torch.tanh(x[mask_middle])
            result[mask_upper] = upper_linear[mask_upper]
            
            # Copy back to original tensor
            x.copy_(result)
            return x
        else:
            # Create masks for different regions
            mask_lower = x <= self.a
            mask_middle = (x > self.a) & (x < self.b)
            mask_upper = x >= self.b
            
            # Initialize result tensor
            result = torch.zeros_like(x)
            
            # Apply different functions to different regions
            result[mask_lower] = lower_linear[mask_lower]
            result[mask_middle] = torch.tanh(x[mask_middle])
            result[mask_upper] = upper_linear[mask_upper]
            
            return result


@register_activation
class BLU(nn.Module):
    r"""
    Applies the Bendable Linear Unit function:

    :math:`\text{BLU}(x) = a \cdot \sqrt{x^2 + 1} - 1 + x`

    where :math:`a \in [-1, 1]` controls the bendability.

    Args:
        a (float, optional): Bendability parameter. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.BLU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.BLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            # Constrain a to be in [-1, 1]
            self.a = nn.Parameter(Tensor([a]).clamp(-1, 1))
        else:
            self.a = Tensor([min(max(a, -1), 1)])  # Clamp a to [-1, 1]

    def forward(self, x) -> Tensor:
        result = self.a * torch.sqrt(x**2 + 1) - 1 + x
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ReBLU(nn.Module):
    r"""
    Applies the Rectified Bendable Linear Unit function:

    :math:`\text{ReBLU}(x) = \begin{cases} 
        a \cdot \sqrt{x^2 + 1} - 1 + x, & x > 0 \\
        0, & x \leq 0 
    \end{cases}`

    where :math:`a \in [-1, 1]` controls the bendability.

    Args:
        a (float, optional): Bendability parameter. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ReBLU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReBLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            # Constrain a to be in [-1, 1]
            self.a = nn.Parameter(Tensor([a]).clamp(-1, 1))
        else:
            self.a = Tensor([min(max(a, -1), 1)])  # Clamp a to [-1, 1]

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x <= 0
            x[~mask] = self.a * torch.sqrt(x[~mask]**2 + 1) - 1 + x[~mask]
            x[mask] = 0
            return x
        else:
            positive_part = self.a * torch.sqrt(x**2 + 1) - 1 + x
            return torch.where(x > 0, positive_part, torch.zeros_like(x))


@register_activation
class DELU(nn.Module):
    r"""
    Applies the DELU activation function:

    :math:`\text{DELU}(x) = \begin{cases} 
        (a + 0.5) \cdot x + |\exp(-x) - 1|, & x \geq 0 \\
        x \cdot \sigma(x), & x < 0 
    \end{cases}`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Scaling parameter. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.DELU(a=0.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            # Positive part: (a + 0.5) * x + |exp(-x) - 1|
            x[~mask] = (self.a + 0.5) * x[~mask] + torch.abs(torch.exp(-x[~mask]) - 1)
            # Negative part: x * sigmoid(x)
            x[mask] = x[mask] * torch.sigmoid(x[mask])
            return x
        else:
            # Positive part: (a + 0.5) * x + |exp(-x) - 1|
            positive_part = (self.a + 0.5) * x + torch.abs(torch.exp(-x) - 1)
            # Negative part: x * sigmoid(x)
            negative_part = x * torch.sigmoid(x)
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class SCMish(nn.Module):
    r"""
    Applies the Soft Clipping Mish activation function:

    :math:`\text{SC-mish}(x) = \max(0, x \cdot \tanh(\text{softplus}(a \cdot x)))`

    Args:
        a (float, optional): Scaling parameter. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SCMish(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SCMish(a=0.25)  # SCL-mish variant
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SCMish(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # softplus(a*x)
        softplus = F.softplus(self.a * x)
        # tanh(softplus(a*x))
        tanh_softplus = torch.tanh(softplus)
        # x * tanh(softplus(a*x))
        mish = x * tanh_softplus
        # max(0, mish)
        result = torch.clamp(mish, min=0)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class SCSwish(nn.Module):
    r"""
    Applies the Soft Clipping Swish activation function:

    :math:`\text{SC-swish}(x) = \max(0, x \cdot \sigma(x))`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SCSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SCSwish(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        # x * sigmoid(x)
        swish = x * torch.sigmoid(x)
        # max(0, swish)
        result = torch.clamp(swish, min=0)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSwish(nn.Module):
    r"""
    Applies the Parametric Swish activation function:

    :math:`\text{p-swish}(x) = \begin{cases} 
        a \cdot x \cdot \sigma(b \cdot x), & x \leq c \\
        x, & x > c 
    \end{cases}`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Scaling parameter. Default: 1.0
        b (float, optional): Sigmoid scaling parameter. Default: 1.0
        c (float, optional): Threshold parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PSwish(a=1.0, b=1.0, c=0.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSwish(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x <= self.c
            # Swish part: a * x * sigmoid(b * x)
            x[mask] = self.a * x[mask] * torch.sigmoid(self.b * x[mask])
            return x
        else:
            # Swish part: a * x * sigmoid(b * x)
            swish_part = self.a * x * torch.sigmoid(self.b * x)
            return torch.where(x <= self.c, swish_part, x)


@register_activation
class PELU(nn.Module):
    r"""
    Applies the Parametric Exponential Linear Unit function:

    :math:`\text{PELU}(x) = \begin{cases} 
        \frac{a}{b} \cdot x, & x \geq 0 \\
        a \cdot \left(\exp\left(\frac{x}{b}\right) - 1\right), & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scaling parameter. Default: 1.0
        b (float, optional): Exponential parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PELU(a=1.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            # Ensure a and b are positive
            self.a = nn.Parameter(Tensor([abs(a)]))
            self.b = nn.Parameter(Tensor([abs(b)]))
        else:
            self.a = Tensor([abs(a)])
            self.b = Tensor([abs(b)])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            # Positive part: (a/b) * x
            x[~mask] = (self.a / self.b) * x[~mask]
            # Negative part: a * (exp(x/b) - 1)
            x[mask] = self.a * (torch.exp(x[mask] / self.b) - 1)
            return x
        else:
            # Positive part: (a/b) * x
            positive_part = (self.a / self.b) * x
            # Negative part: a * (exp(x/b) - 1)
            negative_part = self.a * (torch.exp(x / self.b) - 1)
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class EDELU(nn.Module):
    r"""
    Applies the Extended Exponential Linear Unit function:

    :math:`\text{EDELU}(x) = \begin{cases} 
        x, & x \geq c \\
        \frac{\exp(a \cdot x) - 1}{b}, & x < c 
    \end{cases}`

    where :math:`b \cdot c = \exp(a \cdot c) - 1` to ensure continuity at :math:`x = c`.

    Args:
        a (float, optional): Exponential parameter. Default: 1.0
        c (float, optional): Threshold parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.EDELU(a=1.0, c=0.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.EDELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        c: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        # Calculate b to ensure continuity at x = c
        # b*c = exp(a*c) - 1 => b = (exp(a*c) - 1) / c
        # Handle the case where c is close to zero
        if abs(self.c.item()) < 1e-6:
            b = self.a  # lim_{c->0} (exp(a*c) - 1) / c = a
        else:
            b = (torch.exp(self.a * self.c) - 1) / self.c
        
        if self.inplace:
            mask = x < self.c
            # Negative part: (exp(a*x) - 1) / b
            x[mask] = (torch.exp(self.a * x[mask]) - 1) / b
            return x
        else:
            # Negative part: (exp(a*x) - 1) / b
            negative_part = (torch.exp(self.a * x) - 1) / b
            return torch.where(x >= self.c, x, negative_part)


@register_activation
class AdaptiveCombination1(nn.Module):
    # TODO: Naming
    r"""
    :note: This is a temporary naming.
    Applies an adaptive combination of activation functions:

    :math:`\text{AdaptiveCombination1}(x) = a \cdot \text{LReLU}(x) + (1 - a) \cdot \text{ELU}(x)`

    Args:
        a (float, optional): Mixing parameter. Default: 0.5
        lrelu_slope (float, optional): Negative slope for LReLU. Default: 0.01
        elu_alpha (float, optional): Alpha parameter for ELU. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.AdaptiveCombination1(a=0.7)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AdaptiveCombination1(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.5, 
        lrelu_slope: float = 0.01,
        elu_alpha: float = 1.0,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.lrelu_slope = lrelu_slope
        self.elu_alpha = elu_alpha
        if learnable:
            # Constrain a to be in [0, 1]
            self.a = nn.Parameter(Tensor([a]).clamp(0, 1))
        else:
            self.a = Tensor([min(max(a, 0), 1)])  # Clamp a to [0, 1]

    def forward(self, x) -> Tensor:
        # LReLU(x)
        lrelu = F.leaky_relu(x, negative_slope=self.lrelu_slope)
        
        # ELU(x)
        elu = F.elu(x, alpha=self.elu_alpha)
        
        # Weighted combination
        result = self.a * lrelu + (1 - self.a) * elu
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class AdaptiveCombination2(nn.Module):

    r"""
    :note: This is a temporary naming.
    Applies an adaptive combination of activation functions with sigmoid gating:

    :math:`\text{AdaptiveCombination2}(x) = \sigma(a \cdot x) \cdot \text{PReLU}(x) + (1 - \sigma(a \cdot x)) \cdot \text{PELU}(x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Gating parameter. Default: 1.0
        prelu_slope (float, optional): Negative slope for PReLU. Default: 0.01
        pelu_alpha (float, optional): Alpha parameter for PELU. Default: 1.0
        pelu_beta (float, optional): Beta parameter for PELU. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.AdaptiveCombination2(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AdaptiveCombination2(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        prelu_slope: float = 0.01,
        pelu_alpha: float = 1.0,
        pelu_beta: float = 1.0,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.prelu_slope = prelu_slope
        self.pelu_alpha = pelu_alpha
        self.pelu_beta = pelu_beta
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # Sigmoid gate
        gate = torch.sigmoid(self.a * x)
        
        # PReLU(x)
        prelu = torch.where(x >= 0, x, self.prelu_slope * x)
        
        # PELU(x)
        pelu_pos = (self.pelu_alpha / self.pelu_beta) * x
        pelu_neg = self.pelu_alpha * (torch.exp(x / self.pelu_beta) - 1)
        pelu = torch.where(x >= 0, pelu_pos, pelu_neg)
        
        # Gated combination
        result = gate * prelu + (1 - gate) * pelu
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class FELU(nn.Module):
    r"""
    Applies the Fast Exponential Linear Unit function:

    :math:`\text{FELU}(x) = \begin{cases} 
        x, & x \geq 0 \\
        a \cdot 2^{\frac{x}{\ln(2)}} - 1, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.FELU(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.FELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            # Using 2^(x/ln(2)) = exp(x) for negative values
            x[mask] = self.a * (torch.exp(x[mask]) - 1)
            return x
        else:
            # Using 2^(x/ln(2)) = exp(x) for negative values
            return torch.where(x >= 0, x, self.a * (torch.exp(x) - 1))


@register_activation
class PFELU(nn.Module):
    r"""
    Applies the P+FELU function:

    :math:`\text{P+FELU}(x) = \begin{cases} 
        x + b, & x \geq 0 \\
        a \cdot 2^{\frac{x}{\ln(2)}} - 1 + b, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        b (float, optional): Bias parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PFELU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PFELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            # Using 2^(x/ln(2)) = exp(x) for negative values
            x[mask] = self.a * (torch.exp(x[mask]) - 1) + self.b
            x[~mask] = x[~mask] + self.b
            return x
        else:
            # Using 2^(x/ln(2)) = exp(x) for negative values
            negative_part = self.a * (torch.exp(x) - 1) + self.b
            positive_part = x + self.b
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class MPELU(nn.Module):
    r"""
    Applies the Multiple Parametric Exponential Linear Unit function:

    :math:`\text{MPELU}(x) = \begin{cases} 
        x, & x \geq 0 \\
        a \cdot (\exp(b \cdot x) - 1), & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        b (float, optional): Exponential parameter for negative values. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.MPELU(a=1.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MPELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = self.a * (torch.exp(self.b * x[mask]) - 1)
            return x
        else:
            return torch.where(x >= 0, x, self.a * (torch.exp(self.b * x) - 1))


@register_activation
class PE2ReLU(nn.Module):
    r"""
    Applies the P-E2-ReLU function:

    :math:`\text{P-E2-ReLU}(x) = a \cdot \text{ReLU}(x) + b \cdot \text{ELU}(x) + (1 - a - b) \cdot (-\text{ELU}(-x))`

    Args:
        a (float, optional): Weight for ReLU component. Default: 0.4
        b (float, optional): Weight for ELU component. Default: 0.3
        elu_alpha (float, optional): Alpha parameter for ELU. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PE2ReLU(a=0.4, b=0.3)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PE2ReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.4, 
        b: float = 0.3, 
        elu_alpha: float = 1.0,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.elu_alpha = elu_alpha
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # ReLU(x)
        relu_part = F.relu(x)
        
        # ELU(x)
        elu_part = F.elu(x, alpha=self.elu_alpha)
        
        # -ELU(-x)
        neg_elu_part = -F.elu(-x, alpha=self.elu_alpha)
        
        # Weighted combination
        result = self.a * relu_part + self.b * elu_part + (1 - self.a - self.b) * neg_elu_part
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PE2Id(nn.Module):
    r"""
    Applies the P-E2-Id function:

    :math:`\text{P-E2-Id}(x) = a \cdot x + (1 - a) \cdot (\text{ELU}(x) - \text{ELU}(-x))`

    Args:
        a (float, optional): Weight for identity component. Default: 0.5
        elu_alpha (float, optional): Alpha parameter for ELU. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PE2Id(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PE2Id(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.5, 
        elu_alpha: float = 1.0,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.elu_alpha = elu_alpha
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # Identity part
        id_part = x
        
        # ELU(x) - ELU(-x)
        elu_diff = F.elu(x, alpha=self.elu_alpha) - F.elu(-x, alpha=self.elu_alpha)
        
        # Weighted combination
        result = self.a * id_part + (1 - self.a) * elu_diff
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class SoftExponential(nn.Module):
    r"""
    Applies the Soft Exponential activation function:

    :math:`\text{SoftExponential}(x) = \begin{cases} 
        \frac{\exp(x) - 1}{a} + a, & a > 0 \\
        x, & a = 0 \\
        \frac{\ln(1 - a(x + a))}{-a}, & a < 0 
    \end{cases}`

    Args:
        a (float, optional): Shape parameter. Default: 0.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SoftExponential(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SoftExponential(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        a = self.a.item()  # Get scalar value for conditional logic
        
        if abs(a) < 1e-6:  # a ≈ 0
            return x
        
        if a > 0:
            result = (torch.exp(a * x) - 1) / a + a
        else:  # a < 0
            # Ensure the argument to log is positive
            safe_x = torch.clamp(1 - a * (x + a), min=1e-6)
            result = torch.log(safe_x) / -a
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class CELU(nn.Module):
    r"""
    Applies the Continuously Differentiable ELU function:

    :math:`\text{CELU}(x) = \begin{cases} 
        x, & x \geq 0 \\
        a \cdot \left(\exp\left(\frac{x}{a}\right) - 1\right), & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.CELU(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.CELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = self.a * (torch.exp(x[mask] / self.a) - 1)
            return x
        else:
            return torch.where(x >= 0, x, self.a * (torch.exp(x / self.a) - 1))


@register_activation
class ErfReLU(nn.Module):
    r"""
    Applies the Erf-based ReLU function:

    :math:`\text{ErfReLU}(x) = \begin{cases} 
        x, & x \geq 0 \\
        a \cdot \text{erf}(x), & x < 0 
    \end{cases}`

    where :math:`\text{erf}(x)` is the error function.

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ErfReLU(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ErfReLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[mask] = self.a * torch.erf(x[mask])
            return x
        else:
            return torch.where(x >= 0, x, self.a * torch.erf(x))


@register_activation
class PSELU(nn.Module):
    r"""
    Applies the Parametric Scaled Exponential Linear Unit function:

    :math:`\text{PSELU}(x) = \begin{cases} 
        a \cdot x, & x \geq 0 \\
        a \cdot b \cdot (\exp(x) - 1), & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Scale parameter for negative values. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PSELU(a=1.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask] = self.a * x[~mask]
            x[mask] = self.a * self.b * (torch.exp(x[mask]) - 1)
            return x
        else:
            positive_part = self.a * x
            negative_part = self.a * self.b * (torch.exp(x) - 1)
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class LPSELU(nn.Module):
    r"""
    Applies the Leaky Parametric Scaled Exponential Linear Unit function:

    :math:`\text{LPSELU}(x) = \begin{cases} 
        a \cdot x, & x \geq 0 \\
        a \cdot b \cdot (\exp(x) - 1) + c \cdot x, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Scale parameter for exponential term. Default: 1.0
        c (float, optional): Scale parameter for linear term in negative region. Default: 0.01
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.LPSELU(a=1.0, b=1.0, c=0.01)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LPSELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 0.01, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask] = self.a * x[~mask]
            x[mask] = self.a * self.b * (torch.exp(x[mask]) - 1) + self.c * x[mask]
            return x
        else:
            positive_part = self.a * x
            negative_part = self.a * self.b * (torch.exp(x) - 1) + self.c * x
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class LPSELU_RP(nn.Module):
    r"""
    Applies the Leaky Parametric Scaled Exponential Linear Unit with Reposition Parameter function:

    :math:`\text{LPSELU\_RP}(x) = \begin{cases} 
        a \cdot x + m, & x \geq 0 \\
        a \cdot b \cdot (\exp(x) - 1) + c \cdot x + m, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Scale parameter for exponential term. Default: 1.0
        c (float, optional): Scale parameter for linear term in negative region. Default: 0.01
        m (float, optional): Reposition parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.LPSELU_RP(a=1.0, b=1.0, c=0.01, m=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LPSELU_RP(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 0.01, 
        m: float = 0.0,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
            self.m = nn.Parameter(Tensor([m]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])
            self.m = Tensor([m])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask] = self.a * x[~mask] + self.m
            x[mask] = self.a * self.b * (torch.exp(x[mask]) - 1) + self.c * x[mask] + self.m
            return x
        else:
            positive_part = self.a * x + self.m
            negative_part = self.a * self.b * (torch.exp(x) - 1) + self.c * x + self.m
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class ShELU(nn.Module):
    r"""
    Applies the Shifted ELU (horizontal) function:

    :math:`\text{ShELU}(x) = \begin{cases} 
        x + b, & x + b \geq 0 \\
        a \cdot (\exp(x + b) - 1), & x + b < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        b (float, optional): Horizontal shift parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ShELU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ShELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        shifted_x = x + self.b
        
        if self.inplace:
            mask = shifted_x < 0
            x[~mask] = shifted_x[~mask]
            x[mask] = self.a * (torch.exp(shifted_x[mask]) - 1)
            return x
        else:
            return torch.where(shifted_x >= 0, shifted_x, self.a * (torch.exp(shifted_x) - 1))


@register_activation
class SvELU(nn.Module):
    r"""
    Applies the Shifted ELU (vertical) function:

    :math:`\text{SvELU}(x) = \begin{cases} 
        x + b, & x \geq 0 \\
        a \cdot (\exp(x) - 1) + b, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        b (float, optional): Vertical shift parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SvELU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SvELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask] = x[~mask] + self.b
            x[mask] = self.a * (torch.exp(x[mask]) - 1) + self.b
            return x
        else:
            positive_part = x + self.b
            negative_part = self.a * (torch.exp(x) - 1) + self.b
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class PShELU(nn.Module):
    r"""
    Applies the PELU with Horizontal Shift function:

    :math:`\text{PShELU}(x) = \begin{cases} 
        \frac{a}{b} \cdot (x + c), & x + c \geq 0 \\
        a \cdot \left(\exp\left(\frac{x + c}{b}\right) - 1\right), & x + c < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Divisive parameter. Default: 1.0
        c (float, optional): Horizontal shift parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PShELU(a=1.0, b=1.0, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PShELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        shifted_x = x + self.c
        
        if self.inplace:
            mask = shifted_x < 0
            x[~mask] = (self.a / self.b) * shifted_x[~mask]
            x[mask] = self.a * (torch.exp(shifted_x[mask] / self.b) - 1)
            return x
        else:
            positive_part = (self.a / self.b) * shifted_x
            negative_part = self.a * (torch.exp(shifted_x / self.b) - 1)
            return torch.where(shifted_x >= 0, positive_part, negative_part)


@register_activation
class PSvELU(nn.Module):
    r"""
    Applies the PELU with Vertical Shift function:

    :math:`\text{PSvELU}(x) = \begin{cases} 
        \frac{a}{b} \cdot x + c, & x \geq 0 \\
        a \cdot \left(\exp\left(\frac{x}{b}\right) - 1\right) + c, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Divisive parameter. Default: 1.0
        c (float, optional): Vertical shift parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PSvELU(a=1.0, b=1.0, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSvELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            x[~mask] = (self.a / self.b) * x[~mask] + self.c
            x[mask] = self.a * (torch.exp(x[mask] / self.b) - 1) + self.c
            return x
        else:
            positive_part = (self.a / self.b) * x + self.c
            negative_part = self.a * (torch.exp(x / self.b) - 1) + self.c
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class TSwish(nn.Module):
    r"""
    Applies the Tunable Swish function:

    :math:`\text{T-swish}(x) = \begin{cases} 
        x, & x \geq c \\
        a \cdot x \cdot \sigma(b \cdot x), & x < c 
    \end{cases}`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Sigmoid parameter. Default: 1.0
        c (float, optional): Threshold parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TSwish(a=1.0, b=1.0, c=0.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TSwish(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < self.c
            x[mask] = self.a * x[mask] * torch.sigmoid(self.b * x[mask])
            return x
        else:
            swish_part = self.a * x * torch.sigmoid(self.b * x)
            return torch.where(x >= self.c, x, swish_part)


@register_activation
class RePSU(nn.Module):
    r"""
    Applies the Rectified Parametric Sigmoid Unit function:

    :math:`\text{RePSU}(x) = a \cdot \text{RePSKU}(x) + (1 - a) \cdot \text{RePSHU}(x)`

    where:
    
    :math:`\text{RePSKU}(x) = \begin{cases} 
        \frac{x - b}{1 + \exp\left(-\text{sgn}(x - c) \frac{|x - c|^{d}}{e}\right)}, & x \geq b \\
        0, & x < b 
    \end{cases}`
    
    :math:`\text{RePSHU}(x) = \begin{cases} 
        2x - \text{RePSKU}(x), & x \geq b \\
        0, & x < b 
    \end{cases}`

    Args:
        a (float, optional): Mixing parameter. Default: 0.5
        b (float, optional): Threshold parameter. Default: 0.0
        c (float, optional): Shift parameter. Default: 0.0
        d (float, optional): Power parameter. Default: 1.0
        e (float, optional): Scale parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.RePSU(a=0.5, b=0.0, c=0.0, d=1.0, e=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RePSU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.5, 
        b: float = 0.0, 
        c: float = 0.0, 
        d: float = 1.0, 
        e: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
            self.d = nn.Parameter(Tensor([d]))
            self.e = nn.Parameter(Tensor([e]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])
            self.d = Tensor([d])
            self.e = Tensor([e])

    def _repsku(self, x):
        # Only compute for x >= b
        mask = x >= self.b
        result = torch.zeros_like(x)
        
        if mask.any():
            x_masked = x[mask]
            diff = x_masked - self.c
            sign = torch.sign(diff)
            power_term = torch.pow(torch.abs(diff), self.d) / self.e
            denom = 1 + torch.exp(-sign * power_term)
            result[mask] = (x_masked - self.b) / denom
            
        return result

    def forward(self, x) -> Tensor:
        # Calculate RePSKU
        repsku = self._repsku(x)
        
        # Calculate RePSHU
        repshu = torch.zeros_like(x)
        mask = x >= self.b
        repshu[mask] = 2 * x[mask] - repsku[mask]
        
        # Combine
        result = self.a * repsku + (1 - self.a) * repshu
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PDELU(nn.Module):
    r"""
    Applies the Parametric Deformable Exponential Linear Unit function:

    :math:`\text{PDELU}(x) = \begin{cases} 
        x, & x \geq 0 \\
        a \cdot \left[1 + (1 - b) \cdot x\right]^{\frac{1}{1 - b}} - 1, & x < 0 
    \end{cases}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Deformation parameter. Default: 0.9
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PDELU(a=1.0, b=0.9)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PDELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.9, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Constrain b to avoid division by zero
            self.b = nn.Parameter(Tensor([b]).clamp(0, 0.999))
        else:
            self.a = Tensor([a])
            # Constrain b to avoid division by zero
            self.b = Tensor([min(max(b, 0), 0.999)])

    def forward(self, x) -> Tensor:
        if self.inplace:
            mask = x < 0
            if mask.any():
                base = 1 + (1 - self.b) * x[mask]
                # Ensure base is positive to avoid complex numbers
                base = torch.clamp(base, min=1e-6)
                power = 1 / (1 - self.b)
                x[mask] = self.a * torch.pow(base, power) - 1
            return x
        else:
            positive_part = x
            
            # Calculate negative part
            negative_mask = x < 0
            negative_part = torch.zeros_like(x)
            if negative_mask.any():
                base = 1 + (1 - self.b) * x[negative_mask]
                # Ensure base is positive to avoid complex numbers
                base = torch.clamp(base, min=1e-6)
                power = 1 / (1 - self.b)
                negative_part[negative_mask] = self.a * torch.pow(base, power) - 1
            
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class EELU(nn.Module):
    r"""
    Applies the Elastic Exponential Linear Unit function:

    :math:`\text{EELU}(x) = \begin{cases} 
        k \cdot x, & x \geq 0 \\
        a \cdot (\exp(b \cdot x) - 1), & x < 0 
    \end{cases}`

    where :math:`k \sim \text{truncated } N(1, \sigma^2)` and :math:`\sigma \sim U(0, \epsilon)`.

    Args:
        a (float, optional): Scale parameter for negative values. Default: 1.0
        b (float, optional): Exponential parameter for negative values. Default: 1.0
        epsilon (float, optional): Upper bound for uniform distribution. Default: 0.5
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.EELU(a=1.0, b=1.0, epsilon=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.EELU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        epsilon: float = 0.5, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.epsilon = epsilon
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Sample sigma from uniform distribution
        sigma = torch.rand(1, device=x.device) * self.epsilon
        
        # Sample k from truncated normal distribution
        # We'll approximate truncated normal by sampling and clamping
        k = torch.randn_like(x) * sigma + 1.0
        k = torch.clamp(k, min=0.0)  # Truncate at 0
        
        if self.inplace:
            mask = x < 0
            x[~mask] = k[~mask] * x[~mask]
            x[mask] = self.a * (torch.exp(self.b * x[mask]) - 1)
            return x
        else:
            positive_part = k * x
            negative_part = self.a * (torch.exp(self.b * x) - 1)
            return torch.where(x >= 0, positive_part, negative_part)


@register_activation
class PFPLUS(nn.Module):
    r"""
    Applies the Parametric First Power Linear Unit with Sign function:

    :math:`\text{PFPLUS}(x) = a \cdot x \cdot (1 - b \cdot x)^{H(x) - 1}`

    where :math:`H(x) = \begin{cases} 1, & x \geq 0 \\ 0, & x < 0 \end{cases}` is the Heaviside step function.

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Shape parameter. Default: 0.1
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PFPLUS(a=1.0, b=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PFPLUS(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.1, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Calculate H(x) - 1
        h_minus_1 = torch.where(x >= 0, torch.zeros_like(x), torch.ones_like(x) * (-1))
        
        # Calculate (1 - b*x)^(H(x) - 1)
        base = 1 - self.b * x
        # Avoid negative bases for fractional powers
        safe_base = torch.where(base > 0, base, torch.ones_like(base) * 1e-6)
        power_term = torch.pow(safe_base, h_minus_1)
        
        # Calculate final result
        result = self.a * x * power_term
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PVLU(nn.Module):
    r"""
    Applies the Parametric Variational Linear Unit function:

    :math:`\text{PVLU}(x) = \max(0, x) + a \cdot \sin(b \cdot x)`

    Args:
        a (float, optional): Amplitude parameter for sine term. Default: 0.1
        b (float, optional): Frequency parameter for sine term. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PVLU(a=0.1, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PVLU(learnable=True, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 0.1, 
        b: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        relu_part = F.relu(x)
        sin_part = self.a * torch.sin(self.b * x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(relu_part + sin_part)
            return x
        else:
            return relu_part + sin_part