import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ShiLU(nn.Module):
    r"""
    Applies the ShiLU activation function:

    :math:`\text{ShiLU}(x) = \alpha \cdot \text{ReLU}(x) + \beta`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        alpha (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        beta (float, optional): Bias term added to the output. Default: 0.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ShiLU.png

    Examples::

        >>> m = torch_activation.ShiLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ShiLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, inplace: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(Tensor([alpha]))
        self.beta = nn.Parameter(Tensor([beta]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            F.relu_(x)
            x.mul_(self.alpha)
            x.add_(self.beta)
            return x
        else:
            return self.alpha * F.relu(x) + self.beta


class CReLU(nn.Module):
    r"""
    Applies the Concatenated Rectified Linear Unit activation function.

    :math:`\text{CReLU}(x) = \text{ReLU}(x) \oplus \text{ReLU}(-x)`

     See: https://doi.org/10.48550/arXiv.1603.05201

    Args:
        dim (int, optional): Dimension along which to concatenate in the output tensor. Default: 1
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, C, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(*, 2C, *)`

    Examples::

        >>> m = torch_activation.CReLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = torch_activation.CReLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, dim: int = 0):
        super(CReLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        return F.relu(torch.cat((x, -x), dim=self.dim))


class ReLUN(nn.Module):
    r"""Applies the element-wise function:

    :math:`\text{ReLUN}(x) = \min(\text{ReLU}(x), n)`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        n (float, optional): Upper bound for the function's output. Default is 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ReLUN.png

    Examples::

        >>> m = torch_activation.ReLUN(n=6.0) # ReLU6
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReLUN(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)

    """

    # TODO: Default to RELU6
    def __init__(self, n: float = 1.0, inplace: bool = False):
        super(ReLUN, self).__init__()
        self.n = nn.Parameter(Tensor([n]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return x.clamp_(0, self.n.item())
        else:
            return torch.clamp(x, 0, self.n.item())


class SquaredReLU(nn.Module):
    r"""
    Applies the element-wise function:

    :math:`\text{SquaredReLU}(x) = \text{ReLU}(x)^2`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

     See: https://arxiv.org/pdf/2109.08668.pdf

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SquaredReLU.png

    Examples::

        >>> m = torch_activation.SquaredReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SquaredReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return F.relu_(x).pow_(2)
        else:
            return F.relu(x).pow(2)


class CoLU(nn.Module):
    r"""
    Applies the Collapsing Linear Unit activation function:

    :math:`\text{CoLU}(x) = \frac{x}{1-x \cdot e^{-(x + e^x)}}`

     See: https://doi.org/10.48550/arXiv.2112.12078

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CoLU.png

    Examples::

        >>> m = nn.CoLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.CoLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False):
        super(CoLU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return x.div_(1 - x * torch.exp(-1 * (x + torch.exp(x))))
        else:
            return x / (1 - x * torch.exp(-1 * (x + torch.exp(x))))


class SinLU(nn.Module):
    r"""
    Applies the Sinu-sigmoidal Linear Unit activation function:

    :math:`\text{SinLU}(x) = (x + a \cdot \sin (b \cdot x)) \sigma (x)`

     See: https://doi.org/10.3390/math10030337

    Args:
        a (float, optional): Initial value for sine function magnitude. Default: 1.0.
        b (float, optional): Initial value for sine function period. Default: 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SinLU.png

    Examples::

        >>> m = nn.SinLU(a=5.0, b=6.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, inplace: bool = False):
        super(SinLU, self).__init__()
        self.alpha = nn.Parameter(Tensor([a]))
        self.beta = nn.Parameter(Tensor([b]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        result = x + self.alpha * torch.sin(self.beta * x)
        result *= torch.sigmoid(x)
        return result

    def _forward_inplace(self, x):
        s_x = torch.sigmoid(x)
        x.add_(self.alpha * torch.sin(self.beta * x))
        x.mul_(s_x)
        return x


class CosLU(nn.Module):
    r"""
    Applies the Cosine Linear Unit function:

    :math:`\text{CosLU}(x) = (x + a \cdot \cos(b \cdot x)) \cdot \sigma(x)`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        a (float, optional): Scaling factor for the cosine term. Default is 1.0.
        b (float, optional): Frequency factor for the cosine term. Default is 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CosLU.png

    Examples::

        >>> m = CosLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = CosLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, inplace: bool = False):
        super(CosLU, self).__init__()
        self.alpha = nn.Parameter(Tensor([a]))
        self.beta = nn.Parameter(Tensor([b]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        result = x + self.alpha * torch.cos(self.beta * x)
        result *= torch.sigmoid(x)
        return result

    def _forward_inplace(self, x):
        s_x = torch.sigmoid(x)
        x.add_(self.alpha * torch.cos(self.beta * x))
        x.mul_(s_x)
        return x


class GCU(nn.Module):
    r"""
    Applies the Growing Cosine Unit activation function:

    :math:`\text{GCU}(x) = x \cos (x)`

     See: https://doi.org/10.48550/arXiv.2108.12943

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GCU.png

    Examples::

        >>> m = nn.GCU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.GCU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(GCU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return x.mul_(torch.cos(x))
        else:
            return x * torch.cos(x)
