import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from . import register_activation

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
