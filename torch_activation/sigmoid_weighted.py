import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


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
