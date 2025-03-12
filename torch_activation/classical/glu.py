import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_activation.base import BaseActivation
from torch import Tensor
from typing import Tuple

import torch_activation as tac
from torch_activation import register_activation
from torch_activation.utils import split

class GLU(BaseActivation):
    r"""
    Applies the Gated Linear Unit function:

    :math:`\text{GLU}(z, z') = z \otimes \sigma(z')`

    where :math:`\sigma` is the sigmoid function and :math:`\otimes` is element-wise multiplication.

    Args:
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = GLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        return F.glu(x, dim=self.dim)


class GTU(BaseActivation):
    r"""
    Applies the Gated Tanh Unit function:

    :math:`\text{GTU}(z, z') = \tanh(z) \otimes \sigma(z')`

    where :math:`\sigma` is the sigmoid function and :math:`\otimes` is element-wise multiplication.

    Args:
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = GTU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return torch.tanh(a) * torch.sigmoid(b)


class GReLU(BaseActivation):
    r"""
    Applies the Gated ReLU function:

    :math:`\text{GatedReLU}(z, z') = z \otimes \text{ReLU}(z')`

    where :math:`\otimes` is element-wise multiplication.

    Args:
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = GReLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return a * F.relu(b)


class GEGLU(BaseActivation):
    r"""
    Applies the Gated GELU function:

    :math:`\text{GatedGELU}(z, z') = z \otimes \text{GELU}(z')`

    where :math:`\otimes` is element-wise multiplication.

    Args:
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = GEGLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return a * F.gelu(b)


class SwiGLU(BaseActivation):
    r"""
    Applies the Swish-GELU function:

    :math:`\text{SwiGLU}(z, z') = z \otimes \text{swish}(z')`

    where :math:`\text{swish}(x) = x \cdot \sigma(x)` and :math:`\otimes` is element-wise multiplication.

    Args:
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = SwiGLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return a * tac.Swish()(b)

