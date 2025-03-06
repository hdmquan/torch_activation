import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_activation import register_activation

@register_activation    
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


@register_activation
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


@register_activation
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

    Examples::

        >>> m = nn.SlReLU(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SlReLU(a=2.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
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


@register_activation
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

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/CReLU.png

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


@register_activation
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


@register_activation
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


@register_activation
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


@register_activation
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


@register_activation
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

    Here is a plot of the function:

    .. image:: ../images/activation_images/VLU.png


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

@register_activation
class LReLU(nn.Module):
    r"""
    Applies the Leaky ReLU activation function.

    .. math::
        \text{LReLU}(z) = 
        \begin{cases} 
        z, & z \geq 0, \\
        \frac{z}{a}, & z < 0,
        \end{cases}

    where :math:`a` is recommended to be 100.

    Args:
        a (float, optional): The denominator for negative inputs. Default: ``100.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LReLU.png

    Examples::

        >>> m = torch_activation.LReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LReLU(a=50.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 100.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.div_(self.a))
        else:
            return torch.where(x >= 0, x, x / self.a)


class OLReLU(nn.Module):
    r"""
    Applies the Optimized Leaky ReLU activation function.

    .. math::
        \text{OLReLU}(z) = 
        \begin{cases} 
        z, & z \geq 0, \\
        z \cdot \exp(-a), & z < 0,
        \end{cases}

    where :math:`a = \frac{u+l}{u-l}`.

    Args:
        l (float, optional): Lower bound parameter. Default: ``3.0``
        u (float, optional): Upper bound parameter. Default: ``8.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/OLReLU.png

    Examples::

        >>> m = torch_activation.OLReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.OLReLU(l=2.0, u=6.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, l: float = 3.0, u: float = 8.0, inplace: bool = False):
        super().__init__()
        assert l < u, "Lower bound must be less than upper bound"
        self.a = (u + l) / (u - l)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.mul_(torch.exp(-self.a)))
        else:
            return torch.where(x >= 0, x, x * torch.exp(-self.a))


@register_activation
class RReLU(nn.Module):
    r"""
    Applies the Randomized Leaky ReLU activation function.

    .. math::
        \text{RReLU}(z_i) = 
        \begin{cases} 
        z_i, & z_i \geq 0, \\
        z_i a_i, & z_i < 0,
        \end{cases}

    where :math:`a_i` is sampled from a uniform distribution :math:`U(l, u)`,
    with recommended values :math:`U(3, 8)`.

    Args:
        l (float, optional): Lower bound of the uniform distribution. Default: ``3.0``
        u (float, optional): Upper bound of the uniform distribution. Default: ``8.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RReLU.png

    Examples::

        >>> m = torch_activation.RReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RReLU(l=2.0, u=6.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, l: float = 3.0, u: float = 8.0, inplace: bool = False):
        super().__init__()
        assert 0 < l < u, "Ensure 0 < l < u for the uniform distribution bounds."
        self.l = l
        self.u = u
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        # Sample a_i from U(l, u)
        a = torch.empty_like(x).uniform_(self.l, self.u)
        
        if self.inplace:
            return x.where(x >= 0, x.mul_(a))
        else:
            return torch.where(x >= 0, x, x * a)


@register_activation
class SRReLU(nn.Module):
    r"""
    The Softsign Randomized Leaky ReLU (S-RReLU) is defined as:

    .. math::
        \text{S-RReLU}(z_i) = 
        \begin{cases} 
        z_i, & z_i \geq 0, \\
        \frac{z_i}{a_i}, & z_i < 0,
        \end{cases}

    where :math:`a_i` is sampled for each epoch and neuron i from the uniform distribution
    :math:`a_i \sim U(l, u)` where :math:`l < u` and :math:`l, u \in (0, \infty)`.
    
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    
    Args:
        l (float, optional): Lower bound of the uniform distribution (default: 1/8).
        u (float, optional): Upper bound of the uniform distribution (default: 1/3).
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SRReLU.png

    Examples::

        >>> m = nn.SRReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SRReLU(l=1/4, u=1/2, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, l: float = 1/8, u: float = 1/3, inplace: bool = False):
        super().__init__()
        assert 0 < l < u, "Ensure 0 < l < u for the uniform distribution bounds."
        self.l = l
        self.u = u
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        # Sample a_i from U(l, u)
        a = torch.empty_like(x).uniform_(self.l, self.u)
        
        if self.inplace:
            return x.where(x >= 0, x.div_(a))
        else:
            return torch.where(x >= 0, x, x / a)


@register_activation
class NReLU(nn.Module):
    r"""
    Applies the Noisy ReLU activation function.

    .. math::
        \text{NReLU}(z) = \max(0, z + a)

    where :math:`a \sim N(0, \sigma(z))` is sampled from a Gaussian distribution.

    Args:
        sigma (float, optional): Standard deviation for the noise. Default: ``0.1``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/NReLU.png

    Examples::

        >>> m = torch_activation.NReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.NReLU(sigma=0.2, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, sigma: float = 0.1, inplace: bool = False):
        super().__init__()
        self.sigma = sigma
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        # Generate noise from a normal distribution
        noise = torch.randn_like(x) * self.sigma
        
        if self.inplace:
            x.add_(noise)
            return F.relu_(x)
        else:
            return F.relu(x + noise)
        
class SCAA(nn.Module):
    r"""
    Applies the Spatial Context-Aware Activation function:

    :math:`\text{SCAA}(X) = \max(X, f_{DW}(X))`

    where :math:`f_{DW}` is a depthwise convolution operation.

    Args:
        channels (int): Number of input channels
        kernel_size (int, optional): Size of the convolving kernel. Default: ``3``
        padding (int, optional): Padding added to all sides of the input. Default: ``1``

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        - Output: Same shape as the input

    Examples::

        >>> m = torch_activation.SCAA(channels=64)
        >>> x = torch.randn(1, 64, 28, 28)
        >>> output = m(x)
    """

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=channels,
            bias=False
        )
        # Initialize weights
        nn.init.kaiming_normal_(self.dw_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        return torch.maximum(x, self.dw_conv(x))


@register_activation
class RTReLU(nn.Module):
    r"""
    Applies the Randomly Translational ReLU activation function:

    .. math::
        \text{RT-ReLU}(z_i) = 
        \begin{cases} 
        z_i + a_i, & z_i + a_i \geq 0, \\
        0, & z_i + a_i < 0,
        \end{cases}

    where :math:`a_i \sim N(0, \sigma^2)` is sampled from a Gaussian distribution.

    Args:
        sigma (float, optional): Standard deviation for the Gaussian noise. Default: ``0.75``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RTReLU.png

    Examples::

        >>> m = torch_activation.RTReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RTReLU(sigma=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, sigma: float = 0.75, inplace: bool = False):
        super().__init__()
        self.sigma = sigma
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        # Generate random translations from a normal distribution
        a = torch.randn_like(x) * self.sigma
        
        if self.inplace:
            x.add_(a)
            return F.relu_(x)
        else:
            return F.relu(x + a)


@register_activation
class NLReLU(nn.Module):
    r"""
    Applies the Natural-Logarithm-ReLU activation function:

    :math:`\text{NLReLU}(z) = \ln(a \cdot \max(0, z) + 1)`

    Args:
        a (float, optional): Scaling factor for the ReLU output. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/NLReLU.png

    Examples::

        >>> m = torch_activation.NLReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.NLReLU(a=2.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x = F.relu_(x)
            x.mul_(self.a).add_(1.0).log_()
            return x
        else:
            return torch.log(self.a * F.relu(x) + 1.0)


@register_activation
class SLU(nn.Module):
    r"""
    Applies the Softplus Linear Unit activation function:

    .. math::
        \text{SLU}(z) = 
        \begin{cases} 
        az, & z \geq 0, \\
        b \ln(\exp(z) + 1) - c, & z < 0,
        \end{cases}

    which simplifies to:

    .. math::
        \text{SLU}(z) = 
        \begin{cases} 
        z, & z \geq 0, \\
        2 \ln(\frac{\exp(z) + 1}{2}), & z < 0,
        \end{cases}

    where :math:`a=1, b=2, c=2\ln(2)`.

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SLU.png

    Examples::

        >>> m = torch_activation.SLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.c = 2 * torch.log(torch.tensor(2.0))
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            neg_mask = x < 0
            x[neg_mask] = 2 * torch.log((torch.exp(x[neg_mask]) + 1) / 2)
            return x
        else:
            return torch.where(
                x >= 0,
                x,
                2 * torch.log((torch.exp(x) + 1) / 2)
            )


@register_activation
class ReSP(nn.Module):
    r"""
    Applies the Rectified Softplus activation function:

    .. math::
        \text{ReSP}(z) = 
        \begin{cases} 
        az + \ln(2), & z \geq 0, \\
        \ln(1 + \exp(z)), & z < 0,
        \end{cases}

    Args:
        a (float, optional): Scaling factor for positive inputs. Default: ``1.7``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ReSP.png

    Examples::

        >>> m = torch_activation.ReSP()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReSP(a=1.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.7, inplace: bool = False):
        super().__init__()
        self.a = a
        self.ln2 = torch.log(torch.tensor(2.0))
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            neg_mask = x < 0
            x[~neg_mask] = self.a * x[~neg_mask] + self.ln2
            x[neg_mask] = torch.log(1 + torch.exp(x[neg_mask]))
            return x
        else:
            return torch.where(
                x >= 0,
                self.a * x + self.ln2,
                torch.log(1 + torch.exp(x))
            )


@register_activation
class PReNU(nn.Module):
    r"""
    Applies the Parametric Rectified Non-linear Unit activation function:

    .. math::
        \text{PReNU}(z) = 
        \begin{cases} 
        z - a \ln(z + 1), & z \geq 0, \\
        0, & z < 0,
        \end{cases}

    Args:
        a (float, optional): Parameter controlling the logarithmic term. Default: ``0.25``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PReNU.png

    Examples::

        >>> m = torch_activation.PReNU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PReNU(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.25, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x = F.relu_(x)
            x.sub_(self.a * torch.log(x + 1))
            return x
        else:
            relu_x = F.relu(x)
            return relu_x - self.a * torch.log(relu_x + 1)


@register_activation
class BReLU(nn.Module):
    r"""
    Applies the Bounded ReLU activation function:

    .. math::
        \text{BReLU}(z) = \min(\max(0, z), a) = 
        \begin{cases} 
        0, & z \leq 0, \\
        z, & 0 < z < a, \\
        a, & z \geq a,
        \end{cases}

    Args:
        a (float, optional): Upper bound for the function's output. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/BReLU.png

    Examples::

        >>> m = torch_activation.BReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.BReLU(a=6.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.clamp_(0, self.a)
        else:
            return torch.clamp(x, 0, self.a)


@register_activation
class HardSigmoid(nn.Module):
    r"""
    Applies the Hard Sigmoid activation function:

    :math:`\text{HardSigmoid}(z) = \max(0, \min(\frac{z+1}{2}, 1))`

    or alternatively:

    :math:`\text{HardSigmoid}(z) = \max(0, \min(0.2z + 0.5, 1))`

    Args:
        version (str, optional): Version of hard sigmoid to use ('v1' or 'v2'). Default: ``'v1'``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.HardSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HardSigmoid(version='v2', inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, version: str = 'v1', inplace: bool = False):
        super().__init__()
        assert version in ['v1', 'v2'], "version must be 'v1' or 'v2'"
        self.version = version
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.version == 'v1':
            if self.inplace:
                x.add_(1).div_(2).clamp_(0, 1)
                return x
            else:
                return torch.clamp((x + 1) / 2, 0, 1)
        else:  # v2
            if self.inplace:
                x.mul_(0.2).add_(0.5).clamp_(0, 1)
                return x
            else:
                return torch.clamp(0.2 * x + 0.5, 0, 1)


@register_activation
class HardTanh(nn.Module):
    r"""
    Applies the HardTanh activation function:

    .. math::
        \text{HardTanh}(z) = 
        \begin{cases} 
        a, & z < a, \\
        z, & a \leq z \leq b, \\
        b, & z > b,
        \end{cases}

    Args:
        a (float, optional): Lower bound of the linear region. Default: ``-1.0``
        b (float, optional): Upper bound of the linear region. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardTanh.png

    Examples::

        >>> m = torch_activation.HardTanh()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HardTanh(a=-2.0, b=2.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = -1.0, b: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.b = b
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.clamp_(self.a, self.b)
        else:
            return torch.clamp(x, self.a, self.b)


@register_activation
class SvHardTanh(nn.Module):
    r"""
    Applies the Shifted HardTanh activation function:

    .. math::
        \text{SvHardTanh}(z) = 
        \begin{cases} 
        -1 + a, & z < -1, \\
        z + a, & -1 \leq z \leq 1, \\
        1 + a, & z > 1,
        \end{cases}

    Args:
        a (float, optional): Shift parameter. Default: ``0.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SvHardTanh.png

    Examples::

        >>> m = torch_activation.SvHardTanh()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SvHardTanh(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.clamp_(-1, 1).add_(self.a)
            return x
        else:
            return torch.clamp(x, -1, 1) + self.a
        
@register_activation
class ShHardTanh(nn.Module):
    r"""
    Applies the Shifted HardTanh activation function:

    .. math::
        \text{ShHardTanh}(z) = 
        \begin{cases} 
        -1, & z < -1 - a, \\
        z, & -1 - a \leq z \leq 1 - a, \\
        1, & z > 1 - a,
        \end{cases}

    Args:
        a (float, optional): Shift parameter. Default: ``0.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ShHardTanh.png

    Examples::

        >>> m = torch_activation.ShHardTanh()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ShHardTanh(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.clamp_(-1 - self.a, 1 - self.a).clamp_(-1, 1)
        else:
            return torch.clamp(torch.clamp(x, -1 - self.a, 1 - self.a), -1, 1)


@register_activation
class HardSwish(nn.Module):
    r"""
    Applies the Hard Swish activation function:

    .. math::
        \text{Hard swish}(z) = z \cdot 
        \begin{cases} 
        0, & z \leq -3, \\
        1, & z \geq 3, \\
        \frac{z}{6} + \frac{1}{2}, & -3 < z < 3,
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardSwish.png

    Examples::

        >>> m = torch_activation.HardSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HardSwish(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            inner = x.add_(3).clamp_(0, 6).div_(6)
            x.mul_(inner)
            return x
        else:
            inner = torch.clamp(x + 3, 0, 6) / 6
            return x * inner


@register_activation
class TRec(nn.Module):
    r"""
    Applies the Truncated Rectified activation function:

    .. math::
        \text{TRec}(z) = 
        \begin{cases} 
        z, & z > a, \\
        0, & z \leq a,
        \end{cases}

    Args:
        a (float, optional): Threshold parameter. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TRec.png

    Examples::

        >>> m = torch_activation.TRec()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TRec(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.masked_fill_(x <= self.a, 0)
        else:
            return torch.where(x > self.a, x, torch.zeros_like(x))


@register_activation
class Hardshrink(nn.Module):
    r"""
    Applies the Hardshrink activation function:

    .. math::
        \text{Hardshrink}(z) = 
        \begin{cases} 
        z, & z > a, \\
        0, & -a \leq z \leq a, \\
        z, & z < -a,
        \end{cases}

    where :math:`a > 0`.

    Args:
        a (float, optional): Threshold parameter. Default: ``0.5``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Hardshrink.png

    Examples::

        >>> m = torch_activation.Hardshrink()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Hardshrink(a=1.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.5, inplace: bool = False):
        super().__init__()
        assert a > 0, "Threshold parameter 'a' must be positive"
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            mask = (x >= -self.a) & (x <= self.a)
            x.masked_fill_(mask, 0)
            return x
        else:
            return torch.where((x >= -self.a) & (x <= self.a), torch.zeros_like(x), x)


@register_activation
class Softshrink(nn.Module):
    r"""
    Applies the Softshrink activation function:

    .. math::
        \text{Softshrink}(z) = 
        \begin{cases} 
        z - a, & z > a, \\
        0, & -a \leq z \leq a, \\
        z + a, & z < -a,
        \end{cases}

    where :math:`a > 0`.

    Args:
        a (float, optional): Threshold parameter. Default: ``0.5``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Softshrink.png

    Examples::

        >>> m = torch_activation.Softshrink()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Softshrink(a=1.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.5, inplace: bool = False):
        super().__init__()
        assert a > 0, "Threshold parameter 'a' must be positive"
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            mask_pos = x > self.a
            mask_neg = x < -self.a
            mask_mid = ~(mask_pos | mask_neg)
            
            x[mask_pos] -= self.a
            x[mask_neg] += self.a
            x[mask_mid] = 0
            return x
        else:
            return torch.where(
                x > self.a,
                x - self.a,
                torch.where(
                    x < -self.a,
                    x + self.a,
                    torch.zeros_like(x)
                )
            )


@register_activation
class BLReLU(nn.Module):
    r"""
    Applies the Bounded Leaky ReLU activation function:

    .. math::
        \text{BLReLU}(z) = 
        \begin{cases} 
        az, & z \leq 0, \\
        z, & 0 < z < b, \\
        az + c, & z \geq b,
        \end{cases}

    where :math:`c = (1 - a)b`.

    Args:
        a (float, optional): Slope parameter for negative and large positive inputs. Default: ``0.1``
        b (float, optional): Upper bound of the linear region. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/BLReLU.png

    Examples::

        >>> m = torch_activation.BLReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.BLReLU(a=0.2, b=2.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.1, b: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.b = b
        self.c = (1 - a) * b
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            mask_neg = x <= 0
            mask_pos_large = x >= self.b
            mask_mid = ~(mask_neg | mask_pos_large)
            
            x[mask_neg] *= self.a
            x[mask_pos_large] = self.a * x[mask_pos_large] + self.c
            return x
        else:
            return torch.where(
                x <= 0,
                self.a * x,
                torch.where(
                    x >= self.b,
                    self.a * x + self.c,
                    x
                )
            )


@register_activation
class VReLU(nn.Module):
    r"""
    Applies the V-shaped ReLU activation function:

    .. math::
        \text{vReLU}(z) = |z| = 
        \begin{cases} 
        z, & z \geq 0, \\
        -z, & z < 0,
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/VReLU.png

    Examples::

        >>> m = torch_activation.VReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.VReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.abs_()
            return x
        else:
            return torch.abs(x)


@register_activation
class PanFunction(nn.Module):
    r"""
    Applies the Pan activation function:

    .. math::
        \text{Pan function}(z) = 
        \begin{cases} 
        z - a, & z \geq a, \\
        0, & -a < z < a, \\
        -z - a, & z \leq -a,
        \end{cases}

    Args:
        a (float, optional): Threshold parameter. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PanFunction.png

    Examples::

        >>> m = torch_activation.PanFunction()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PanFunction(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            mask_pos = x >= self.a
            mask_neg = x <= -self.a
            mask_mid = ~(mask_pos | mask_neg)
            
            x[mask_pos] -= self.a
            x[mask_neg] = -x[mask_neg] - self.a
            x[mask_mid] = 0
            return x
        else:
            return torch.where(
                x >= self.a,
                x - self.a,
                torch.where(
                    x <= -self.a,
                    -x - self.a,
                    torch.zeros_like(x)
                )
            )


@register_activation
class AbsLU(nn.Module):
    r"""
    Applies the Absolute Linear Unit activation function:

    .. math::
        \text{AbsLU}(z) = 
        \begin{cases} 
        z, & z \geq 0, \\
        a|z|, & z < 0,
        \end{cases}

    where :math:`a \in [0, 1]`.

    Args:
        a (float, optional): Scaling parameter for negative inputs. Default: ``0.5``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/AbsLU.png

    Examples::

        >>> m = torch_activation.AbsLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AbsLU(a=0.2, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.5, inplace: bool = False):
        super().__init__()
        assert 0 <= a <= 1, "Parameter 'a' must be in the range [0, 1]"
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            mask_neg = x < 0
            x[mask_neg] = self.a * x[mask_neg].abs_()
            return x
        else:
            return torch.where(x >= 0, x, self.a * torch.abs(x))


@register_activation
class MReLU(nn.Module):
    r"""
    Applies the Mirrored Rectified Linear Unit activation function:

    .. math::
        \text{mReLU}(z) = \min(\text{ReLU}(1 - z), \text{ReLU}(1 + z)) = 
        \begin{cases} 
        1 + z, & -1 \leq z \leq 0, \\
        1 - z, & 0 < z \leq 1, \\
        0, & \text{otherwise},
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MReLU.png

    Examples::

        >>> m = torch_activation.MReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            # Cannot be done fully in-place
            return torch.minimum(F.relu(1 - x), F.relu(1 + x))
        else:
            return torch.minimum(F.relu(1 - x), F.relu(1 + x))


@register_activation
class LSPTLU(nn.Module):
    r"""
    Applies the Linear Symmetric Piecewise Triangular Linear Unit activation function:

    .. math::
        \text{LSPTLU}(z) = 
        \begin{cases} 
        0.2z, & z < 0, \\
        z, & 0 \leq z \leq a, \\
        2a - z, & a < z \leq 2a, \\
        0, & z > 2a,
        \end{cases}

    Args:
        a (float, optional): Parameter controlling the shape. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LSPTLU.png

    Examples::

        >>> m = torch_activation.LSPTLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LSPTLU(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super().__init__()
        self.a = a
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            mask_neg = x < 0
            mask_mid = (0 <= x) & (x <= self.a)
            mask_high = (self.a < x) & (x <= 2 * self.a)
            mask_very_high = x > 2 * self.a
            
            x[mask_neg] *= 0.2
            # No change for mask_mid (0 <= x <= a)
            x[mask_high] = 2 * self.a - x[mask_high]
            x[mask_very_high] = 0
            return x
        else:
            return torch.where(
                x < 0,
                0.2 * x,
                torch.where(
                    x <= self.a,
                    x,
                    torch.where(
                        x <= 2 * self.a,
                        2 * self.a - x,
                        torch.zeros_like(x)
                    )
                )
            )
        
if __name__ == "__main__":
    from torch_activation.utils import plot_activation

    shifted_relu_p = {}
    softsign_rrelu_p = {"l": [1/8, 1/4], "u": [1/5, 1/2]}
    slrelu_p = {"a": [2, 10]}
    crelu_p = {}
    relun_p = {"n": [1, 6]}
    squared_relu_p = {}
    sine_relu_p = {"a": [0.5, 2]}
    minsin_p = {}
    vlu_p = {"a": [0.5, 2], "b": [0.5, 2]}
    lrelu_p = {"a": [50, 100]}
    rrelu_p = {"l": [2, 3], "u": [6, 8]}
    srrelu_p = {"l": [1/8, 1/4], "u": [1/5, 1/2]}
    nrelu_p = {"sigma": [0.05, 0.2]}
    rtrelu_p = {"sigma": [0.5, 1.0]}
    nlrelu_p = {"a": [0.5, 1, 2]}
    slu_p = {}
    resp_p = {"a": [1.5, 2.0]}
    prenu_p = {"a": [0.1, 0.5]}
    brelu_p = {"a": [1, 3, 6]}
    hard_sigmoid_p = {"version": ["v1", "v2"]}
    hard_tanh_p = {"a": [-2, -1], "b": [1, 2]}
    sv_hard_tanh_p = {"a": [0, 0.5, 1]}
    sh_hard_tanh_p = {"a": [0, 0.5, 1]}
    hard_swish_p = {}
    trec_p = {"a": [0.5, 1, 2]}
    hardshrink_p = {"a": [0.5, 1, 2]}
    softshrink_p = {"a": [0.5, 1, 2]}
    blrelu_p = {"a": [0.1, 0.2], "b": [1, 2]}
    vrelu_p = {}
    pan_function_p = {"a": [0.5, 2]}
    abslu_p = {"a": [0.2, 0.8]}
    mrelu_p = {}
    lsptlu_p = {"a": [0.5, 1, 2]}
    
    plot_activation(ShiftedReLU, shifted_relu_p)
    # plot_activation(SoftsignRReLU, softsign_rrelu_p)
    plot_activation(SlReLU, slrelu_p)
    plot_activation(CReLU, crelu_p)
    plot_activation(ReLUN, relun_p)
    plot_activation(SquaredReLU, squared_relu_p)
    plot_activation(SineReLU, sine_relu_p)
    plot_activation(Minsin, minsin_p)
    plot_activation(VLU, vlu_p)
    plot_activation(LReLU, lrelu_p)
    plot_activation(RReLU, rrelu_p)
    plot_activation(SRReLU, srrelu_p)
    plot_activation(NReLU, nrelu_p)
    plot_activation(RTReLU, rtrelu_p)
    plot_activation(NLReLU, nlrelu_p)
    plot_activation(SLU, slu_p)
    plot_activation(ReSP, resp_p)
    plot_activation(PReNU, prenu_p)
    plot_activation(BReLU, brelu_p)
    # plot_activation(HardSigmoid, hard_sigmoid_p)
    plot_activation(HardTanh, hard_tanh_p)
    plot_activation(SvHardTanh, sv_hard_tanh_p)
    plot_activation(ShHardTanh, sh_hard_tanh_p)
    plot_activation(HardSwish, hard_swish_p)
    plot_activation(TRec, trec_p)
    plot_activation(Hardshrink, hardshrink_p)
    plot_activation(Softshrink, softshrink_p)
    plot_activation(BLReLU, blrelu_p)
    plot_activation(VReLU, vrelu_p)
    plot_activation(PanFunction, pan_function_p)
    plot_activation(AbsLU, abslu_p)
    plot_activation(MReLU, mrelu_p)
    plot_activation(LSPTLU, lsptlu_p)
    
    # NOTE: SCAA is not included as it's not a one-to-one function
    # and requires specific input dimensions