import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_activation.base import BaseActivation

from torch import Tensor
from typing import Callable

from torch_activation import register_activation

@register_activation
class SGT(BaseActivation):
    r"""
    Applies the SGT activation function:

    :math:`\text{SGT}(x) = \begin{cases} az_i^{b_i}, z_i \geq 0 \\cz_i^{d_i}, z_i < 0 \end{cases}`

     See: https://www.nature.com/articles/s41598-022-19020-y

    Args:
        a (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        b (float, optional): Exponent for the positive part of the input. Default: 1.0.
        c (float, optional): Scaling factor for the negative part of the input. Default: 1.0.
        d (float, optional): Exponent for the negative part of the input. Default: 1.0.
        learnable (bool, optional): optionally make b and d parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SGT.png

    Examples::

        >>> m = torch_activation.SGT(a=2.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SGT(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 1.0, 
        c: float = 1.0, 
        d: float = 1.0, 
        learnable: bool = False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.a = a
        self.c = c
        
        if learnable:
            self.b = nn.Parameter(Tensor([b]))
            self.d = nn.Parameter(Tensor([d]))
        else:
            self.b = Tensor([b])
            self.d = Tensor([d])

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = x < 0
        
        if self.inplace:
            x_pos = x.clone()
            x_pos[neg_mask] = 0
            x_neg = x.clone()
            x_neg[pos_mask] = 0
            
            x_pos[pos_mask] = self.a * torch.pow(x_pos[pos_mask], self.b)
            x_neg[neg_mask] = self.c * torch.pow(x_neg[neg_mask], self.d)
            
            x.copy_(x_pos + x_neg)
            return x
        else:
            result = torch.zeros_like(x)
            result[pos_mask] = self.a * torch.pow(x[pos_mask], self.b)
            result[neg_mask] = self.c * torch.pow(x[neg_mask], self.d)
            
            return result
