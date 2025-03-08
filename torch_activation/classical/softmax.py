import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation

@register_activation
class Softmax(nn.Module):
    r"""
    Applies the Softmax function:

    :math:`\text{Softmax}(z_j) = \frac{\exp(z_j)}{\sum_{k=1}^{N} \exp(z_k)}`

    where :math:`z_j` is the input of neuron j in a softmax layer consisting of N neurons.

    Args:
        dim (int, optional): A dimension along which Softmax will be computed. Default: -1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Softmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    def __init__(self, dim=-1, inplace=False):
        super(Softmax, self).__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.softmax_(dim=self.dim)
            return x
        else:
            return F.softmax(x, dim=self.dim)


@register_activation
class BetaSoftmax(nn.Module):
    r"""
    Applies the β-Softmax function:

    :math:`\text{β-Softmax}(z_j) = \frac{\exp(β \cdot z_j)}{\sum_{k=1}^{N} \exp(β \cdot z_k)}`

    where :math:`z_j` is the input of neuron j in a softmax layer consisting of N neurons,
    and :math:`β` is a trainable or fixed parameter controlling the sharpness of the distribution.

    Args:
        beta (float, optional): Initial value for the beta parameter. Default: 1.0
        trainable (bool, optional): If True, beta is a trainable parameter. Default: False
        dim (int, optional): A dimension along which Softmax will be computed. Default: -1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = BetaSoftmax(beta=2.0, trainable=True)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    def __init__(self, beta=1.0, trainable=False, dim=-1, inplace=False):
        super(BetaSoftmax, self).__init__()
        self.dim = dim
        self.inplace = inplace
        
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x = x.mul_(self.beta)
            x.softmax_(dim=self.dim)
            return x
        else:
            scaled_x = self.beta * x
            return F.softmax(scaled_x, dim=self.dim)
