from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


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
        learnable (bool, optional): optionally make alpha and beta parameters trainable. Default: ``False``
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

        if learnable:
            self.alpha = nn.Parameter(Tensor([alpha]))
            self.beta = nn.Parameter(Tensor([beta]))
        else:
            self.alpha = Tensor([alpha])
            self.beta = Tensor([beta])

    def _forward(self, x) -> Tensor:
        alpha = self.alpha.to(x.dtype) if isinstance(self.alpha, Tensor) else self.alpha
        beta = self.beta.to(x.dtype) if isinstance(self.beta, Tensor) else self.beta
        pos_mask = x >= 0
        neg_mask = x < 0

        result = torch.zeros_like(x)
        result[neg_mask] = self.a * torch.pow(x[neg_mask], alpha)
        result[pos_mask] = self.b * torch.pow(x[pos_mask], beta)

        return result

    def _forward_inplace(self, x) -> Tensor:
        alpha = self.alpha.to(x.dtype) if isinstance(self.alpha, Tensor) else self.alpha
        beta = self.beta.to(x.dtype) if isinstance(self.beta, Tensor) else self.beta
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        if neg_mask.any():
            x[neg_mask] = self.a * torch.pow(x[neg_mask], alpha)

        if pos_mask.any():
            x[pos_mask] = self.b * torch.pow(x[pos_mask], beta)

        return x


if __name__ == "__main__":
    from torch_activation.utils import plot_activation

    activation_params = {"SGT": {"a": [1, 2], "b": [1, 2], "alpha": [1, 2], "beta": [1, 2]}}

    for activation_name, params in activation_params.items():
        # Get the class from its name
        activation_class = globals()[activation_name]
        plot_activation(activation_class, params)
