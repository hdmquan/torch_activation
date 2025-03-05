import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from . import register_activation

@register_activation
class CosLU(nn.Module):
    r"""
    Applies the Cosine Linear Unit function:

    :math:`\text{CosLU}(x) = (x + a \cdot \cos(b \cdot x)) \cdot \sigma(x)`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        a (float, optional): Scaling factor for the cosine term. Default is 1.0.
        b (float, optional): Frequency factor for the cosine term. Default is 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CosLU.png

    Examples::

        >>> m = CosLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = CosLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, inplace: bool = False):
        super(CosLU, self).__init__()
        self.alpha = nn.Parameter(Tensor([a]))
        self.beta = nn.Parameter(Tensor([b]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        result = x + self.alpha * torch.cos(self.beta * x)
        result *= torch.sigmoid(x)
        return result

    def _forward_inplace(self, x):
        s_x = torch.sigmoid(x)
        x.add_(self.alpha * torch.cos(self.beta * x))
        x.mul_(s_x)
        return x
