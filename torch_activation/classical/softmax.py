import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class Softmax(BaseActivation):
    r"""
    Applies the Softmax function:

    :math:`\text{Softmax}(z_j) = \frac{\exp(z_j)}{\sum_{k=1}^{N} \exp(z_k)}`

    Args:
        dim (int, optional): a dimension along which Softmax will be computed. Default: ``-1``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Softmax.png

    Examples::

        >>> m = torch_activation.Softmax(dim=1)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, dim=-1, inplace=False, **kwargs):
        super().__init__(inplace=inplace, **kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.softmax_(dim=self.dim)
            return x
        else:
            return F.softmax(x, dim=self.dim)


@register_activation
class BetaSoftmax(BaseActivation):
    r"""
    Applies the β-Softmax function:

    :math:`\text{β-Softmax}(z_j) = \frac{\exp(\beta \cdot z_j)}{\sum_{k=1}^{N} \exp(\beta \cdot z_k)}`

    Args:
        beta (float, optional): initial value for the beta parameter. Default: ``1.0``
        trainable (bool, optional): if ``True``, beta is a trainable parameter. Default: ``False``
        dim (int, optional): a dimension along which Softmax will be computed. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/BetaSoftmax.png

    Examples::

        >>> m = torch_activation.BetaSoftmax(beta=2.0, trainable=True)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, beta: float = 1.0, trainable: bool = False, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        if trainable:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def _forward(self, x: Tensor) -> Tensor:
        return F.softmax(self.beta * x, dim=self.dim)
