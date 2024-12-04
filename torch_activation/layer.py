import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable


class LinComb(nn.Module):
    r"""
    Applies the LinComb activation function:

    :math:`\text{LinComb}(x) = \sum_{i=1}^{n} w_i \cdot F_i(x)`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        activations (Iterable[nn.Module]): List of activation functions. Default: [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softsign]

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LinComb.png

    Examples::

        >>> activations = [nn.ReLU(), nn.Sigmoid()]
        >>> m = LinComb(activation_functions)
        >>> input = torch.randn(10)
        >>> output = m(input)
    """

    def __init__(
        self,
        activations: Iterable[nn.Module] = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Softsign(),
        ],
    ):
        super(LinComb, self).__init__()
        self.activations = nn.ModuleList(activations)
        self.weights = nn.Parameter(Tensor(len(activations)))

        self.weights.data.uniform_(-1, 1)

    def forward(self, input) -> Tensor:
        activations = [
            self.weights[i] * self.activations[i](input)
            for i in range(len(self.activations))
        ]
        return torch.sum(torch.stack(activations), dim=0)


class NormLinComb(nn.Module):
    r"""
    Applies the LinComb activation function:

    :math:`\text{NormLinComb}(x) = \frac{\sum_{i=1}^{n} w_i \cdot F_i(x)}{\|\|W\|\|}`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        activations (Iterable[nn.Module]): List of activation functions. Default: [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softsign]

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/NormLinComb.png

    Examples::

        >>> activations = [nn.ReLU, nn.Sigmoid]
        >>> m = NormLinComb(activation_functions)
        >>> input = torch.randn(10)
        >>> output = m(input)
    """

    def __init__(
        self,
        activations: Iterable[nn.Module] = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Softsign(),
        ],
    ):
        super(NormLinComb, self).__init__()
        self.activations = nn.ModuleList(activations)
        self.weights = nn.Parameter(torch.Tensor(len(activations)))

        self.weights.data.uniform_(-1, 1)

    def forward(self, input) -> torch.Tensor:
        activations = [
            self.weights[i] * self.activations[i](input)
            for i in range(len(self.activations))
        ]
        output = torch.sum(torch.stack(activations), dim=0)
        return output / torch.norm(output)


# GLUs
class ReGLU(nn.Module):
    r"""
    Applies the GeGLU activation function, defined as:

    :math:`\text{GeGLU}(x) = \text{ReLU} (xW + b) \odot (xV + c)`

     See: https://doi.org/10.48550/arXiv.2002.05202

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = ReGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)

    """

    def __init__(self, dim: int = -1):
        super(ReGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=self.dim)
        return a * F.relu(b)


class GeGLU(nn.Module):
    r"""
    Applies the GeGLU activation function, defined as:

    :math:`\text{GeGLU}(x) = \text{GELU} (xW + b) \odot (xV + c)`

     See: https://doi.org/10.48550/arXiv.2002.05202

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = GeGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, dim: int = -1):
        super(GeGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class SwiGLU(nn.Module):
    r"""
    Applies the SwiGLU activation function, defined as:

    :math:`\sigma(x) =  \text{Swish} (xW + b) \odot (xV + c)`

     See: https://doi.org/10.48550/arXiv.2002.05202

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = SwiGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, dim: int = -1):
        super(SwiGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)


class SeGLU(nn.Module):
    r"""
    Applies the SeGLU activation function, defined as:

    :math:`\text{SeGLU}(x) =  \text{SELU} (xW + b) \odot (xV + c)`

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = SeGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, dim: int = -1):
        super(SeGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.selu(b)
