import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from torch_activation import register_activation

class GLU(nn.Module):
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

    def __init__(self, dim: int = -1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.glu(x, dim=self.dim)


class GTU(nn.Module):
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

    def __init__(self, dim: int = -1):
        super(GTU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return torch.tanh(a) * torch.sigmoid(b)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)


class GReLU(nn.Module):
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

    def __init__(self, dim: int = -1):
        super(GReLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return a * F.relu(b)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)


class GEGLU(nn.Module):
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

    def __init__(self, dim: int = -1):
        super(GEGLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return a * F.gelu(b)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)


class SwiGLU(nn.Module):
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

    def __init__(self, dim: int = -1):
        super(SwiGLU, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = self._split(x)
        return a * (b * torch.sigmoid(b))  # swish(x) = x * sigmoid(x)

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dim_size = x.size(self.dim)
        assert dim_size % 2 == 0, f"Dimension {self.dim} must be divisible by 2"
        
        split_size = dim_size // 2
        return torch.split(x, split_size, dim=self.dim)
