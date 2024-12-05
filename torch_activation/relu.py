import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ShiftedReLU(nn.Module):
    r"""
    A Shifted ReLU is a simple translation of a ReLU and is defined as:


    :math:`\text{ShiftedReLU}(x) = \text{max}(-1, x)`

    See: http://arxiv.org/abs/1511.07289

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ShiftedReLU.png
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Inplace with max? C++?
        if self.inplace:
            return F.relu_(x - 1.0)
        else:
            return F.relu(x - 1.0)


class SoftsignRReLU(nn.Module):
    r"""
    The Softsign Randomized Leaky ReLU (S-RReLU) is defined as:

    .. math::
        `\text{S-RReLU}(z_i) = 
        \begin{cases} 
        \frac{1}{(1+z_i)^2} + z_i, &  z_i \geq 0, \\
        \frac{1}{(1+z_i)^2} + a_i z_i, & z_i < 0,
        \end{cases}`

    where :math:`a_i` is sampled for each epoch and neuron i from the uniform distribution
    :math:`a_i \sim U(l, u)` where :math:`l < u` and :math:`l, u \in (0, \infty)`.

    See: http://dx.doi.org/10.1007/s00521-023-08565-2
    
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    
    Args:
        l (float, optional): Lower bound of the uniform distribution (default: 1/8).
        u (float, optional): Upper bound of the uniform distribution (default: 1/3).
    """

    def __init__(self, l: float = 1 / 8, u: float = 1 / 3):
        super().__init__()
        assert 0 < l < u, "Ensure 0 < l < u for the uniform distribution bounds."
        self.l = l
        self.u = u

    # TODO: There should be a better way to implement this
    def forward(self, x: Tensor) -> Tensor:
        # Sample a_i from U(l, u)
        a = torch.empty_like(x).uniform_(self.l, self.u)

        common_term = 1 / (1 + x).pow(2)

        # Apply the activation function using torch.where
        return torch.where(x >= 0, common_term + x, common_term + a * x)


class SlReLU(nn.Module):
    r"""
    A Sloped ReLU (SlReLU) [242] is similar to the LReLU — whereas the LReLU parameterizes the slope for negative
    inputs, the SlReLU parameterizes the slope of ReLU for positive inputs. It is, therefore, defined as:

    .. math::
        `\text{SlReLU}(z) = 
        \begin{cases} 
        a \cdot z, & z \geq 0, \\
        0, & z < 0,
        \end{cases}` 
        
    a is recommended to be from 1 to 10.
    
    See: https://doi.org/10.1109/pimrc.2017.8292678
    
    Args:
        a (float, optional): The slope for positive inputs. Default: 10.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``
        
    Shape:
        - Input: :math:`(*, C, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(*, 2C, *)`
        
    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SlReLU.png
    """

    def __init__(self, a=10.0, inplace: bool = False):
        super(SlReLU, self).__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return F.relu_(self.a * x)
        else:
            return F.relu(self.a * x)


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


class SineReLU(nn.Module):
    r"""
    Applies the element-wise function:

    .. math::
        \text{SineReLU}(z) = 
        \begin{cases} 
        z, & \text{if } z \geq 0 \\
        a (\sin(z) - \cos(z)), & \text{if } z < 0
        \end{cases}

    Args:
        a (float, optional): The scaling parameter for the negative inputs. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function:

    .. image:: ../images/activation_images/SineReLU.png

    Examples::

        >>> m = torch_activation.SineReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SineReLU(a=0.5)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.mul_(self.a * (torch.sin(x) - torch.cos(x))))
        else:
            return torch.where(x >= 0, x, self.a * (torch.sin(x) - torch.cos(x)))


class Minsin(nn.Module):
    r"""
    Applies the element-wise function:

    .. math::`\text{Minsin}(x) =
        \begin{cases} 
        \sin(x), & \text{if } x \geq 0 \\
        x, & \text{if } x < 0 
        \end{cases}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Here is a plot of the function:

    .. image:: ../images/activation_images/Minsin.png

    Examples::

        >>> m = Minsin()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, inplace: bool = False):
        super(Minsin, self).__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.sin_())
        else:
            return torch.where(x >= 0, torch.sin(x), x)


class VLU(nn.Module):
    r"""
    Applies the element-wise function:

    :math:`\text{VLU}(x) = \text{ReLU}(x) + a \sin(bx) = \max(0, x) + a \sin(bx)`

    Args:
        a (float): Scaling factor for the sine component. Default: ``1.0``
        b (float): Frequency multiplier for the sine component. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = VLU(a=1.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.b = b
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            # TODO: Is this correct?
            return torch.relu_(x) + self.a * torch.sin(self.b * x)
        else:
            return torch.relu(x) + self.a * torch.sin(self.b * x)
