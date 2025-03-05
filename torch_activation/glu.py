import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from . import register_activation

class GatedLinearUnit(nn.Module):
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

        >>> m = GatedLinearUnit()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1):
        super(GatedLinearUnit, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.glu(x, dim=self.dim)


class GatedTanhUnit(nn.Module):
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

        >>> m = GatedTanhUnit()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1):
        super(GatedTanhUnit, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return torch.tanh(a) * torch.sigmoid(b)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)


class GatedReLU(nn.Module):
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

        >>> m = GatedReLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1):
        super(GatedReLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return a * F.relu(b)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)


class GatedGELU(nn.Module):
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

        >>> m = GatedGELU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1):
        super(GatedGELU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return a * F.gelu(b)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)


class SwishGELU(nn.Module):
    r"""
    Applies the Swish-GELU function:

    :math:`\text{SwishGELU}(z, z') = z \otimes \text{swish}(z')`

    where :math:`\text{swish}(x) = x \cdot \sigma(x)` and :math:`\otimes` is element-wise multiplication.

    Args:
        dim (int, optional): The dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` where `*` means any number of dimensions
        - Output: :math:`(*, N/2, *)` where `*` means any number of dimensions

    Examples::

        >>> m = SwishGELU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1):
        super(SwishGELU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return a * (b * torch.sigmoid(b))  # swish(x) = x * sigmoid(x)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)
