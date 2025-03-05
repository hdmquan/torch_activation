import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from . import register_activation

@register_activation
class ScaledSoftSign(torch.nn.Module):
    r"""
    Applies the ScaledSoftSign activation function:

    :math:`\text{ScaledSoftSign}(x) = \frac{\alpha \cdot x}{\beta + \|x\|}`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        alpha (float, optional): The initial value of the alpha parameter. Default: 1.0
        beta (float, optional): The initial value of the beta parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledSoftSign.png

    Examples:
        >>> m = ScaledSoftSign(alpha=0.5, beta=1.0)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = ScaledSoftSign(inplace=True)
        >>> x = torch.randn(2, 3)
        >>> m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):

        super(ScaledSoftSign, self).__init__()

        self.alpha = torch.nn.Parameter(Tensor([alpha]))
        self.beta = torch.nn.Parameter(Tensor([beta]))

    def forward(self, x) -> Tensor:
        abs_x = x.abs()
        alpha_x = self.alpha * x
        denom = self.beta + abs_x
        result = alpha_x / denom
        return result


@register_activation
class GCU(nn.Module):
    r"""
    Applies the Growing Cosine Unit activation function:

    :math:`\text{GCU}(x) = x \cos (x)`

     See: https://doi.org/10.48550/arXiv.2108.12943

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GCU.png

    Examples::

        >>> m = nn.GCU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.GCU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(GCU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return x.mul_(torch.cos(x))
        else:
            return x * torch.cos(x)


"""
class DReLUs(nn.Module):
    
    Applies the DReLUs activation function:

    :math:`\text{DReLUs}(x) = \begin{cases} \alpha (e ^ x - 1), x \leqslant 0 \\x, \text{otherwise} \end{cases}`

    Args:
        alpha (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples:
        >>> m = nn.DReLUs()
        >>> x = torch.randn(2)
        >>> output = m(x)
    
    
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)
        
    def _forward(self, x):
        return torch.where(x > 0, x,
                           self.alpha * (torch.exp(x) - 1))
        
    def _forward_inplace(self, x):
        x[x <= 0] = (torch.exp(x[x <= 0]) - 1) * self.alpha
        return x
 """
