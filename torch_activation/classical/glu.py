import torch
import torch.nn.functional as F
from torch import Tensor

import torch_activation as tac
from torch_activation.base import BaseActivation
from torch_activation.utils import split


class GLU(BaseActivation):
    r"""
    Applies the Gated Linear Unit activation function:

    :math:`\text{GLU}(z, z') = z \otimes \sigma(z')`

     See: https://doi.org/10.48550/arXiv.1612.08083

    Args:
        dim (int, optional): The dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, C/2)`, half the last dimension.

    .. image:: ../images/activation_images/GLU.png

    Examples::

        >>> m = torch_activation.GLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.GLU(dim=0)
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
    Applies the Gated Tanh Unit activation function:

    :math:`\text{GTU}(z, z') = \tanh(z) \otimes \sigma(z')`

    Args:
        dim (int, optional): The dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, C/2)`, half the last dimension.

    .. image:: ../images/activation_images/GTU.png

    Examples::

        >>> m = torch_activation.GTU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.GTU(dim=0)
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
    Applies the Gated ReLU activation function:

    :math:`\text{GReLU}(z, z') = z \otimes \text{ReLU}(z')`

    Args:
        dim (int, optional): The dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, C/2)`, half the last dimension.

    .. image:: ../images/activation_images/GReLU.png

    Examples::

        >>> m = torch_activation.GReLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.GReLU(dim=0)
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
    Applies the Gated GELU activation function:

    :math:`\text{GEGLU}(z, z') = z \otimes \text{GELU}(z')`

     See: https://doi.org/10.48550/arXiv.2002.05202

    Args:
        dim (int, optional): The dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, C/2)`, half the last dimension.

    .. image:: ../images/activation_images/GEGLU.png

    Examples::

        >>> m = torch_activation.GEGLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.GEGLU(dim=0)
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return a * F.gelu(b)


class ReGLU(BaseActivation):
    r"""
    Applies the Rectified Gated Linear Unit activation function:

    :math:`\text{ReGLU}(z, z') = z \otimes \text{ReLU}(z')`

     See: https://doi.org/10.48550/arXiv.2002.05202

    Args:
        dim (int, optional): The dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, C/2)`, half the last dimension.

    .. image:: ../images/activation_images/ReGLU.png

    Examples::

        >>> m = torch_activation.ReGLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.ReGLU(dim=0)
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return a * F.relu(b)


class SwiGLU(BaseActivation):
    r"""
    Applies the Swish-Gated Linear Unit activation function:

    :math:`\text{SwiGLU}(z, z') = z \otimes \text{swish}(z')`

     See: https://doi.org/10.48550/arXiv.2002.05202

    Args:
        dim (int, optional): The dimension on which to split the input. Default: ``-1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*, C/2)`, half the last dimension.

    .. image:: ../images/activation_images/SwiGLU.png

    Examples::

        >>> m = torch_activation.SwiGLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.SwiGLU(dim=0)
        >>> x = torch.randn(4, 2)
        >>> output = m(x)
    """

    def __init__(self, dim: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        a, b = split(x, self.dim)
        return a * tac.Swish()(b)
