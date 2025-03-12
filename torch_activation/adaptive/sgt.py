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

    :math:`\text{SGT}(x) = \begin{cases} ax^{\alpha}, x < 0 \\bx^{\beta}, x \geq 0 \end{cases}`

     See: https://www.nature.com/articles/s41598-022-19020-y

    Args:
        a (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        alpha (float, optional): Exponent for the positive part of the input. Default: 1.0.
        b (float, optional): Scaling factor for the negative part of the input. Default: 1.0.
        beta (float, optional): Exponent for the negative part of the input. Default: 1.0.
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
        a: float = 0.1, 
        alpha: float = 1.0, 
        b: float = 1.1, 
        beta: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.a = a
        self.b = b  
        self.inplace = inplace
        
        if learnable:
            self.alpha = nn.Parameter(Tensor([alpha]))
            self.beta = nn.Parameter(Tensor([beta]))
        else:
            self.alpha = Tensor([alpha])
            self.beta = Tensor([beta])

    def _forward(self, x) -> Tensor:
        pos_mask = x >= 0
        neg_mask = x < 0
        
        if self.inplace:
            x_pos = x.clone()
            x_pos[neg_mask] = 0
            x_neg = x.clone()
            x_pos[pos_mask] = 0
            
            x_neg[neg_mask] = self.a * torch.pow(x_neg[neg_mask], self.alpha)
            x_pos[pos_mask] = self.b * torch.pow(x_pos[pos_mask], self.beta)
            
            x.copy_(x_pos + x_neg)
            return x
        else:
            result = torch.zeros_like(x)
            result[neg_mask] = self.a * torch.pow(x[pos_mask], self.alpha)
            result[pos_mask] = self.b * torch.pow(x[neg_mask], self.beta)
            
            return result
