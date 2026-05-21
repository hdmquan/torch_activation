import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class ReLU(BaseActivation):
    r"""
    Applies the ReLU activation function:

    :math:`\text{ReLU}(x) = \max(0, x)`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ReLU.png

    Examples::

        >>> m = torch_activation.ReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        return F.relu_(x)


@register_activation
class SReLU(BaseActivation):
    r"""
    Applies the SReLU activation function:

    :math:`\text{SReLU}(x) = \max(0, x - 1)`

     See: http://arxiv.org/abs/1511.07289

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SReLU.png

    Examples::

        >>> m = torch_activation.SReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extra_repr(self):
        return "shift=1.0"

    def _forward(self, x: Tensor) -> Tensor:
        return F.relu(x - 1.0)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        return x.sub_(1.0).clamp_(min=0)


@register_activation
class LReLU(BaseActivation):
    r"""
    Applies the Leaky ReLU activation function:

    :math:`\text{LReLU}(x) = \max(0, x) + \alpha \min(0, x)`

    Args:
        alpha (float, optional): The slope for negative inputs. Default: ``0.01``
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

        >>> m = torch_activation.LReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, alpha: float = 0.01, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=self.alpha)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        return F.leaky_relu_(x, negative_slope=self.alpha)


@register_activation
class VLReLU(BaseActivation):
    r"""
    Applies the Very Leaky ReLU activation function:

    :math:`\text{VLReLU}(x) = \max(0, x) + \alpha \min(0, x)`

    Args:
        alpha (float, optional): The slope for negative inputs. Default: ``3.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/VLReLU.png

    Examples::

        >>> m = torch_activation.VLReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.VLReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, alpha: float = 3.0, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=self.alpha)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        return F.leaky_relu_(x, negative_slope=self.alpha)


@register_activation
class RReLU(BaseActivation):
    r"""
    Applies the Randomized Leaky ReLU activation function:

    .. math::
        \text{RReLU}(z_i) =
        \begin{cases}
        z_i, & z_i \geq 0, \\
        z_i a_i, & z_i < 0,
        \end{cases}

    where :math:`a_i` is sampled for each neuron i from the uniform distribution
    :math:`a_i \sim U(l, u)` where :math:`l < u` and :math:`l, u \in (0, \infty)`.

     See: https://arxiv.org/abs/2303.01360

    Args:
        lower (float, optional): Lower bound of the uniform distribution. Default: ``0.125``
        upper (float, optional): Upper bound of the uniform distribution. Default: ``0.333``
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

        >>> m = torch_activation.RReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, lower: float = 0.125, upper: float = 0.333, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def _forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu_(x, negative_slope=torch.rand(x.shape).uniform_(self.lower, self.upper))

    def _forward_inplace(self, x: Tensor) -> Tensor:
        return F.leaky_relu_(x, negative_slope=torch.rand(x.shape).uniform_(self.lower, self.upper))


# FIXME: Does not pass test
@register_activation
class OLReLU(BaseActivation):
    r"""
    Applies the Optimized Leaky ReLU (OLReLU) activation function:

    .. math::
        \text{OLReLU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        z \cdot \exp(-\alpha), & z < 0,
        \end{cases}

    where :math:`\alpha = \frac{u + l}{u - l}` and :math:`u` and :math:`l` are hyperparameters
    of the bounds of the RReLU.

    Args:
        lower (float, optional): Lower bound parameter l. Default: ``0.125``
        upper (float, optional): Upper bound parameter u. Default: ``0.333``
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

        >>> m = torch_activation.OLReLU(lower=0.1, upper=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, lower: float = 0.125, upper: float = 0.333, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

        # Calculate alpha according to the formula in the paper
        self.alpha = (upper + lower) / (upper - lower)
        self.negative_slope = float(torch.exp(torch.tensor(-self.alpha)))

    def _forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        return F.leaky_relu_(x, negative_slope=self.negative_slope)


@register_activation
class SoftsignRReLU(BaseActivation):
    r"""
    Applies the Softsign Randomized Leaky ReLU (SoftsignRReLU) activation function:

    .. math::
        \text{SoftsignRReLU}(z_i) =
        \begin{cases}
        \frac{1}{(1+z_i)^2} + z_i, & z_i \geq 0, \\
        \frac{1}{(1+z_i)^2} + a_i z_i, & z_i < 0,
        \end{cases}

    where :math:`a_i` is sampled for each epoch and neuron i from the uniform distribution
    :math:`a_i \sim U(l, u)` where :math:`l < u` and :math:`l, u \in (0, \infty)`.

     See: http://dx.doi.org/10.1007/s00521-023-08565-2

    Args:
        lower (float, optional): Lower bound of the uniform distribution. Default: ``0.125``
        u (float, optional): Upper bound of the uniform distribution. Default: ``0.333``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SoftsignRReLU.png

    Examples::

        >>> m = torch_activation.SoftsignRReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SoftsignRReLU(lower=0.1, u=0.4)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, lower: float = 1 / 8, u: float = 1 / 3, **kwargs):
        super().__init__(**kwargs)
        assert 0 < lower < u, "Ensure 0 < l < u for the uniform distribution bounds."
        self.lower = lower
        self.u = u

    # TODO: There should be a better way to implement this
    def _forward(self, x: Tensor) -> Tensor:
        if self.training:
            a = torch.empty_like(x).uniform_(self.lower, self.u)
        else:
            a = torch.full_like(x, (self.lower + self.u) / 2)
        denom = (1 + x).pow(2).clamp(min=1e-7)
        common_term = 1 / denom
        return torch.where(x >= 0, common_term + x, common_term + a * x)


@register_activation
class SlReLU(BaseActivation):
    r"""
    Applies the Sloped ReLU (SlReLU) activation function:

    .. math::
        \text{SlReLU}(z_i) =
        \begin{cases}
        \alpha \cdot z_i, & z_i \geq 0, \\
        0, & z_i < 0,
        \end{cases}

    Args:
        alpha (float, optional): The scaling factor for positive inputs. Default: ``10.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SlReLU.png

    Examples::

        >>> m = torch_activation.SlReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SlReLU(alpha=5.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, alpha: float = 10.0, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        # TODO: Performance
        return torch.clamp(self.alpha * x, min=0)


@register_activation
class CReLU(BaseActivation):
    r"""
    Applies the Concatenated Rectified Linear Unit activation function:

    :math:`\text{CReLU}(x) = \text{ReLU}(x) \oplus \text{ReLU}(-x)`

     See: https://doi.org/10.48550/arXiv.1603.05201

    Args:
        dim (int, optional): Dimension along which to concatenate in the output tensor. Default: ``0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, C, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(*, 2C, *)`, doubles the size of the concatenated dimension.

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

    def __init__(self, dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        return F.relu(torch.cat((x, -x), dim=self.dim))


# TODO: BAF mentioned in the same entry
@register_activation
class NCReLU(BaseActivation):
    r"""
    Applies the Negative Concatenated Rectified Linear Unit activation function:

    :math:`\text{NCReLU}(x) = \text{ReLU}(x) \oplus -\text{ReLU}(-x)`

    Args:
        dim (int, optional): Dimension along which to concatenate in the output tensor. Default: ``0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, C, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(*, 2C, *)`, doubles the size of the concatenated dimension.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/NCReLU.png

    Examples::

        >>> m = torch_activation.NCReLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = torch_activation.NCReLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _forward(self, x: Tensor) -> Tensor:
        return torch.cat((F.relu(x), -F.relu(-x)), dim=self.dim)


@register_activation
class ReLUN(BaseActivation):
    r"""
    Applies the ReLUN activation function:

    :math:`\text{ReLUN}(x) = \min(\text{ReLU}(x), n)`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        n (float, optional): Upper bound for the function's output. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ReLUN.png

    Examples::

        >>> m = torch_activation.ReLUN(n=6.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReLUN(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    # TODO: Default to RELU6
    def __init__(self, n: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n = nn.Parameter(Tensor([n]))

    def _forward(self, x: Tensor) -> Tensor:
        return x.clamp(0) - F.relu(x - self.n)


@register_activation
class SquaredReLU(BaseActivation):
    r"""
    Applies the Squared ReLU activation function:

    :math:`\text{SquaredReLU}(x) = \text{ReLU}(x)^2`

     See: https://arxiv.org/pdf/2109.08668.pdf

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return F.relu_(x).pow_(2)
        else:
            return F.relu(x).pow(2)


@register_activation
class SineReLU(BaseActivation):
    r"""
    Applies the SineReLU activation function:

    .. math::
        \text{SineReLU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        a (\sin(z) - \cos(z)), & z < 0,
        \end{cases}

    Args:
        a (float, optional): scaling parameter for negative inputs. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SineReLU.png

    Examples::

        >>> m = torch_activation.SineReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SineReLU(a=0.5)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.mul_(self.a * (torch.sin(x) - torch.cos(x))))
        else:
            return torch.where(x >= 0, x, self.a * (torch.sin(x) - torch.cos(x)))


@register_activation
class Minsin(BaseActivation):
    r"""
    Applies the Minsin activation function:

    .. math::
        \text{Minsin}(x) =
        \begin{cases}
        \sin(x), & x \geq 0, \\
        x, & x < 0,
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Minsin.png

    Examples::

        >>> m = torch_activation.Minsin()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Minsin(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.sin_())
        else:
            return torch.where(x >= 0, torch.sin(x), x)


@register_activation
class VLU(BaseActivation):
    r"""
    Applies the VLU activation function:

    :math:`\text{VLU}(x) = \text{ReLU}(x) + a \sin(bx) = \max(0, x) + a \sin(bx)`

    Args:
        a (float, optional): Scaling factor for the sine component. Default: ``1.0``
        b (float, optional): Frequency multiplier for the sine component. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/VLU.png

    Examples::

        >>> m = torch_activation.VLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.VLU(a=0.5, b=2.0)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _forward(self, x: Tensor) -> Tensor:
        return torch.relu(x) + self.a * torch.sin(self.b * x)


@register_activation
class LReLU(BaseActivation):  # noqa: F811
    r"""
    Applies the Leaky ReLU activation function:

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

    def __init__(self, a: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.where(x >= 0, x.div_(self.a))
        else:
            return torch.where(x >= 0, x, x / self.a)


class OLReLU(BaseActivation):  # noqa: F811
    r"""
    Applies the Optimized Leaky ReLU activation function:

    .. math::
        \text{OLReLU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        z \cdot \exp(-a), & z < 0,
        \end{cases}

    where :math:`a = \frac{u+l}{u-l}`.

    Args:
        lower (float, optional): Lower bound parameter. Default: ``3.0``
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

        >>> m = torch_activation.OLReLU(lower=2.0, u=6.0, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, lower: float = 3.0, u: float = 8.0, **kwargs):
        super().__init__(**kwargs)
        assert lower < u, "Lower bound must be less than upper bound"
        self.a = (u + lower) / (u - lower)

    def _forward(self, x: Tensor) -> Tensor:
        import math

        neg_slope = math.exp(-self.a)
        if self.inplace:
            return x.where(x >= 0, x.mul_(neg_slope))
        else:
            return torch.where(x >= 0, x, x * neg_slope)


@register_activation
class RReLU(BaseActivation):  # noqa: F811
    r"""
    Applies the Randomized Leaky ReLU activation function:

    .. math::
        \text{RReLU}(z_i) =
        \begin{cases}
        z_i, & z_i \geq 0, \\
        z_i a_i, & z_i < 0,
        \end{cases}

    where the negative slope :math:`1/a_i` is derived from :math:`a_i` sampled from
    a uniform distribution :math:`U(l, u)`, with recommended values :math:`U(3, 8)`.

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

    def __init__(self, lower: float = 3.0, u: float = 8.0, **kwargs):
        super().__init__(**kwargs)
        assert 0 < lower < u, "Ensure 0 < l < u for the uniform distribution bounds."
        self.lower = lower
        self.u = u

    def _forward(self, x: Tensor) -> Tensor:
        if self.training:
            a = torch.empty_like(x).uniform_(self.lower, self.u)
        else:
            a = torch.full_like(x, (self.lower + self.u) / 2)

        if self.inplace:
            return x.where(x >= 0, x.div_(a))
        else:
            return torch.where(x >= 0, x, x / a)


@register_activation
class SRReLU(BaseActivation):
    r"""
    Applies the Softsign Randomized Leaky ReLU (SRReLU) activation function:

    .. math::
        \text{SRReLU}(z_i) =
        \begin{cases}
        \frac{1}{(1+z_i)^2} + z_i, & z_i \geq 0, \\
        \frac{1}{(1+z_i)^2} + a_i z_i, & z_i < 0,
        \end{cases}

    where :math:`a_i` is sampled for each epoch and neuron i from the uniform distribution
    :math:`a_i \sim U(l, u)` where :math:`l < u` and :math:`l, u \in (0, \infty)`.

    Args:
        lower (float, optional): Lower bound of the uniform distribution. Default: ``0.125``
        u (float, optional): Upper bound of the uniform distribution. Default: ``0.333``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SRReLU.png

    Examples::

        >>> m = torch_activation.SRReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SRReLU(lower=0.25, u=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, lower: float = 1 / 8, u: float = 1 / 3, **kwargs):
        super().__init__(**kwargs)
        assert 0 < lower < u, "Ensure 0 < l < u for the uniform distribution bounds."
        self.lower = lower
        self.u = u

    def _forward(self, x: Tensor) -> Tensor:
        if self.training:
            a = torch.empty_like(x).uniform_(self.lower, self.u)
        else:
            a = torch.full_like(x, (self.lower + self.u) / 2)
        frac = 1 / torch.square(1 + x).clamp(min=1e-7)
        return torch.where(x >= 0, frac + x, frac + (a * x))


@register_activation
class NReLU(BaseActivation):
    r"""
    Applies the Noisy ReLU (NReLU) activation function:

    :math:`\text{NReLU}(z) = \max(0, z + a)`

    where :math:`a \sim N(0, \sigma(z))` is a stochastic parameter sampled from a Gaussian
    distribution with zero mean and variance :math:`\sigma(z)^2`, and :math:`\sigma(z)` is
    the standard deviation of the inputs :math:`z`.

    Args:
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

        >>> m = torch_activation.NReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                std = torch.std(x)
            noise = torch.randn_like(x) * std
        else:
            noise = torch.zeros_like(x)

        if self.inplace:
            x.add_(noise)
            return F.relu_(x)
        else:
            return F.relu(x + noise)


# TODO: Really really check this again. Should be correct, but I'm not sure.
class SCAA(BaseActivation):
    r"""
    Applies the Spatial Context-Aware Activation function:

    :math:`\text{SCAA}(X) = \max(X, f_{DW}(X))`

    where :math:`f_{DW}` is a depthwise convolution operation.

    Args:
        channels (int): Number of input channels
        kernel_size (int, optional): Size of the convolving kernel. Default: ``3``
        padding (int, optional): Padding added to all sides of the input. Default: ``1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SCAA.png

    Examples::

        >>> m = torch_activation.SCAA(channels=64)
        >>> x = torch.randn(1, 64, 28, 28)
        >>> output = m(x)

        >>> m = torch_activation.SCAA(channels=32, kernel_size=5, padding=2)
        >>> x = torch.randn(1, 32, 16, 16)
        >>> m(x)
    """

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        # Initialize weights
        nn.init.kaiming_normal_(self.dw_conv.weight, mode="fan_out", nonlinearity="relu")

    def _forward(self, x: Tensor) -> Tensor:
        return torch.maximum(x, self.dw_conv(x))


@register_activation
class RTReLU(BaseActivation):
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

    def __init__(self, sigma: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def _forward(self, x: Tensor) -> Tensor:
        if self.training:
            a = torch.randn_like(x) * self.sigma
        else:
            a = torch.zeros_like(x)

        if self.inplace:
            x.add_(a)
            return F.relu_(x)
        else:
            return F.relu(x + a)


@register_activation
class NLReLU(BaseActivation):
    r"""
    Applies the Natural-Logarithm ReLU (NLReLU) activation function:

    :math:`\text{NLReLU}(z) = \ln(a \cdot \max(0, z) + 1)`

    Args:
        a (float, optional): scaling factor for the ReLU output. Default: ``1.0``
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

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.log(self.a * F.relu(x) + 1.0)


@register_activation
class SLU(BaseActivation):
    r"""
    Applies the Softplus Linear Unit (SLU) activation function:

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c = 2 * torch.log(torch.tensor(2.0))

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            neg_mask = x < 0
            x[neg_mask] = 2 * torch.log((torch.exp(x[neg_mask]) + 1) / 2)
            return x
        else:
            # TODO: Performance
            return torch.where(x >= 0, x, 2 * torch.log((torch.exp(x) + 1) / 2))


@register_activation
class ReSP(BaseActivation):
    r"""
    Applies the Rectified Softplus (ReSP) activation function:

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

    def __init__(self, a: float = 1.7, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.ln2 = torch.log(torch.tensor(2.0))

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            neg_mask = x < 0
            x[~neg_mask] = self.a * x[~neg_mask] + self.ln2
            x[neg_mask] = torch.log(1 + torch.exp(x[neg_mask]))
            return x
        else:
            return torch.where(x >= 0, self.a * x + self.ln2, torch.log(1 + torch.exp(x)))


@register_activation
class PReNU(BaseActivation):
    r"""
    Applies the Parametric Rectified Non-linear Unit (PReNU) activation function:

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

    def __init__(self, a: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        # TODO: The mathematical definition does not have this step.
        pos_x = F.relu(x)  # Un-negative first
        return pos_x - self.a * torch.log(pos_x + 1)


@register_activation
class BReLU(BaseActivation):
    r"""
    Applies the Bounded Rectified Linear Unit (BReLU) activation function:

    .. math::
        \text{BReLU}(z) = \min(\max(0, z), a) =
        \begin{cases}
        0, & z \leq 0, \\
        z, & 0 < z < a, \\
        a, & z \geq a,
        \end{cases}

    Args:
        a (float, optional): upper bound for the output. Default: ``1.0``
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

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.clamp_(0, self.a)
        else:
            return torch.clamp(x, 0, self.a)


# NOTE: Hm... version?
@register_activation
class HardSigmoid(BaseActivation):
    r"""
    Applies the Hard Sigmoid activation function:

    :math:`\text{HardSigmoid}(z) = \max(0, \min(\frac{z+1}{2}, 1))`

    Args:
        version (int, optional): Version of hard sigmoid to use (``1`` or ``2``). Default: ``1``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardSigmoid.png

    Examples::

        >>> m = torch_activation.HardSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HardSigmoid(version='v2', inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, version: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert version in [1, 2], "version must be 1 or 2"
        self.version = version

    def _forward(self, x: Tensor) -> Tensor:
        if self.version == 1:
            if self.inplace:
                x.add_(1).div_(2).clamp_(0, 1)
                return x
            else:
                return torch.clamp((x + 1) / 2, 0, 1)
        else:  # 2
            if self.inplace:
                x.mul_(0.2).add_(0.5).clamp_(0, 1)
                return x
            else:
                return torch.clamp(0.2 * x + 0.5, 0, 1)


@register_activation
class HardTanh(BaseActivation):
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

    def __init__(self, a: float = -1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.clamp_(self.a, self.b)
        else:
            return torch.clamp(x, self.a, self.b)


@register_activation
class SvHardTanh(BaseActivation):
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

    def __init__(self, a: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.clamp_(-1, 1).add_(self.a)
            return x
        else:
            return torch.clamp(x, -1, 1) + self.a


@register_activation
class ShHardTanh(BaseActivation):
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

    def __init__(self, a: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        lo, hi = -(1.0 + self.a), 1.0 - self.a
        return torch.where(x < lo, torch.full_like(x, -1.0),
                           torch.where(x > hi, torch.ones_like(x), x))


@register_activation
class HardSwish(BaseActivation):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return F.hardswish_(x)
        else:
            inner = torch.clamp(x + 3, 0, 6) / 6
            return x * inner


@register_activation
class TRec(BaseActivation):
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

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x > self.a, x, torch.zeros_like(x))


@register_activation
class Hardshrink(BaseActivation):
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

    def __init__(self, a: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert a > 0, "Threshold parameter 'a' must be positive"
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where((x >= -self.a) & (x <= self.a), torch.zeros_like(x), x)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask = (x >= -self.a) & (x <= self.a)
        x.masked_fill_(mask, 0)
        return x


@register_activation
class Softshrink(BaseActivation):
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

    def __init__(self, a: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert a > 0, "Threshold parameter 'a' must be positive"
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(
            x > self.a,
            x - self.a,
            torch.where(x < -self.a, x + self.a, torch.zeros_like(x)),
        )

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_pos = x > self.a
        mask_neg = x < -self.a
        mask_mid = ~(mask_pos | mask_neg)

        x[mask_pos] -= self.a
        x[mask_neg] += self.a
        x[mask_mid] = 0
        return x


@register_activation
class BLReLU(BaseActivation):
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
        a (float, optional): Slope parameter for negative and large positive inputs. Default: ``0.1`` # noqa: E501
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

    def __init__(self, a: float = 0.1, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = (1 - a) * b

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x <= 0, self.a * x, torch.where(x >= self.b, self.a * x + self.c, x))

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_neg = x <= 0
        mask_pos_large = x >= self.b

        x[mask_neg] *= self.a
        x[mask_pos_large] = self.a * x[mask_pos_large] + self.c
        return x


@register_activation
class VReLU(BaseActivation):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.abs(x)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        x.abs_()
        return x


@register_activation
class PanFunction(BaseActivation):
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

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(
            x >= self.a,
            x - self.a,
            torch.where(x <= -self.a, -x - self.a, torch.zeros_like(x)),
        )

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_pos = x >= self.a
        mask_neg = x <= -self.a
        mask_mid = ~(mask_pos | mask_neg)

        x[mask_pos] -= self.a
        x[mask_neg] = -x[mask_neg] - self.a
        x[mask_mid] = 0
        return x


@register_activation
class AbsLU(BaseActivation):
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

    def __init__(self, a: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= a <= 1, "Parameter 'a' must be in the range [0, 1]"
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, self.a * torch.abs(x))

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_neg = x < 0
        x[mask_neg] = self.a * x[mask_neg].abs_()
        return x


@register_activation
class mReLU(BaseActivation):
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

    .. image:: ../images/activation_images/mReLU.png

    Examples::

        >>> m = torch_activation.mReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.mReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.minimum(F.relu(1 - x), F.relu(1 + x))


@register_activation
class LSPTLU(BaseActivation):
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

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(
            x < 0,
            0.2 * x,
            torch.where(
                x <= self.a,
                x,
                torch.where(x <= 2 * self.a, 2 * self.a - x, torch.zeros_like(x)),
            ),
        )

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_neg = x < 0
        mask_mid = (0 <= x) & (x <= self.a)
        mask_high = (self.a < x) & (x <= 2 * self.a)
        mask_very_high = x > 2 * self.a

        x[mask_neg] *= 0.2
        x[mask_mid] = x[mask_mid]
        x[mask_high] = 2 * self.a - x[mask_high]
        x[mask_very_high] = 0
        return x


@register_activation
class SoftModulusQ(BaseActivation):
    r"""
    Applies the SoftModulusQ activation function:

    .. math::
        \text{SoftModulusQ}(z) =
        \begin{cases}
        z^2 (2 - |z|), & |z| \leq 1, \\
        |z|, & |z| > 1,
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SoftModulusQ.png

    Examples::

        >>> m = torch_activation.SoftModulusQ()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SoftModulusQ(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        abs_x = torch.abs(x)
        return torch.where(abs_x <= 1, x.pow(2) * (2 - abs_x), abs_x)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        abs_x = torch.abs(x)
        mask = abs_x <= 1
        x[mask] = x[mask].pow(2) * (2 - abs_x[mask])
        x[~mask] = abs_x[~mask]
        return x


@register_activation
class SoftModulusT(BaseActivation):
    r"""
    Applies the SoftModulusT activation function:

    .. math::
        \text{SoftModulusT}(z) = z \cdot \tanh\left(\frac{z}{a}\right)

    where :math:`a` is a predetermined parameter. When :math:`a = 1`, the SoftModulusT becomes
    the LiSHT activation function.

    Args:
        a (float, optional): Parameter controlling the shape. Default: ``0.01``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SoftModulusT.png

    Examples::

        >>> m = torch_activation.SoftModulusT()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SoftModulusT(a=0.1, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(x / self.a)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        x.mul_(torch.tanh(x / self.a))
        return x


@register_activation
class SignReLU(BaseActivation):
    r"""
    Applies the SignReLU activation function:

    .. math::
        \text{SignReLU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        a \frac{z}{|z|+1}, & z < 0,
        \end{cases}

    where :math:`a` is a fixed parameter. The SignReLU becomes ReLU for :math:`a = 0`.
    This function is also sometimes referred to as DLU (Dual Linear Unit) in the literature.

    Args:
        a (float, optional): Parameter controlling the negative part. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SignReLU.png

    Examples::

        >>> m = torch_activation.SignReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SignReLU(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, self.a * (x / (torch.abs(x) + 1)))

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_neg = x < 0
        x[mask_neg] = self.a * x[mask_neg] / (torch.abs(x[mask_neg]) + 1)
        return x


@register_activation
class LiReLU(BaseActivation):
    r"""
    Applies the Li-ReLU activation function:

    .. math::
        \text{Li-ReLU}(z) =
        \begin{cases}
        az + z, & z \geq 0, \\
        az, & z < 0,
        \end{cases}

    where :math:`a` is a fixed parameter.

    Args:
        a (float, optional): Parameter controlling the linear component. Default: ``0.2``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LiReLU.png

    Examples::

        >>> m = torch_activation.LiReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LiReLU(a=0.5, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, self.a * x + x, self.a * x)

    def _forward_inplace(self, x: Tensor) -> Tensor:
        mask_pos = x >= 0
        mask_neg = x < 0
        x[mask_pos] *= 1 + self.a
        x[mask_neg] *= self.a
        return x


# TODO: Cannot be tested currently because it have a weird requirement
# @register_activation
class DualReLU(BaseActivation):
    r"""
    Applies the DualReLU activation function:

    .. math::
        \text{DualReLU}(z, z') = \max(0, z) - \max(0, z') =
        \begin{cases}
        0, & z \leq 0 \land z' \leq 0, \\
        z, & z > 0 \land z' \leq 0, \\
        -b, & z \leq 0 \land z' > 0, \\
        a - b, & z > 0 \land z' > 0,
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, 2, *)` where :math:`*` means any number of dimensions
        - Output: :math:`(*, 1, *)`

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DualReLU.png

    Examples::

        >>> m = torch_activation.DualReLU()
        >>> x = torch.randn(2, 2)
        >>> output = m(x)

        >>> m = torch_activation.DualReLU(inplace=True)
        >>> x = torch.randn(3, 2, 4)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        size_2_dims = [i for i, s in enumerate(x.shape) if s == 2]
        if not size_2_dims:
            raise ValueError("Input tensor must have a dimension with size 2 for DualReLU")

        dim = size_2_dims[0]

        z, z_prime = torch.split(x, 1, dim=dim)

        result = F.relu(z) - F.relu(z_prime)

        return result


# TODO
# @register_activation
class OPLU(BaseActivation):
    r"""
    Applies the Orthogonal Permutation Linear Unit (OPLU) activation function:

    .. math::
        \text{OPLU}(z_i, z_j) =
        \begin{cases}
        \max(z_i, z_j), & \text{neuron } i, \\
        \min(z_i, z_j), & \text{neuron } j,
        \end{cases}

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, 2n, *)` where :math:`*` means any number of dimensions and n is the number of pairs
        - Output: :math:`(*, 2n, *)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/OPLU.png

    Examples::

        >>> m = torch_activation.OPLU()
        >>> x = torch.randn(4, 2)
        >>> output = m(x)

        >>> m = torch_activation.OPLU(inplace=True)
        >>> x = torch.randn(4, 4)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("The last dimension of the input tensor must be even for OPLU")

        shape = x.shape
        x_reshaped = x.view(*shape[:-1], -1, 2)

        max_vals = torch.max(x_reshaped, dim=-1, keepdim=True)[0]
        min_vals = torch.min(x_reshaped, dim=-1, keepdim=True)[0]

        result = torch.cat((max_vals, min_vals), dim=-1)

        return result.view(*shape)


# TODO: Questionable...
@register_activation
class EReLU(BaseActivation):
    r"""
    Applies the Elastic ReLU (EReLU) activation function:

    .. math::
        \text{EReLU}(z_i) =
        \begin{cases}
        k_i z_i, & z_i \geq 0, \\
        0, & z_i < 0,
        \end{cases}

    where :math:`k_i` is sampled for each epoch and neuron i from the uniform distribution
    :math:`k_i \sim U(1 - \alpha, 1 + \alpha)` where :math:`\alpha \in (0, 1)` is a parameter
    controlling the degree of response fluctuations.

    Args:
        alpha (float, optional): Parameter controlling the degree of response fluctuations. Default: ``0.1``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/EReLU.png

    Examples::

        >>> m = torch_activation.EReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.EReLU(alpha=0.2, inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, alpha: float = 0.1, training: bool = True, **kwargs):
        super().__init__(**kwargs)
        assert 0 < alpha < 1, "alpha must be in the range (0, 1)"
        self.alpha = alpha
        self.training = training

    def _forward(self, x: Tensor) -> Tensor:
        # Zero out negative values
        pos_mask = x >= 0
        result = torch.zeros_like(x)

        if pos_mask.any():
            if self.training:
                # Sample k_i from U(1-alpha, 1+alpha) during training
                k = torch.empty_like(x).uniform_(1 - self.alpha, 1 + self.alpha)
                result[pos_mask] = k[pos_mask] * x[pos_mask]
            else:
                # Use expected value E(k_i) = 1 during testing
                result[pos_mask] = x[pos_mask]

        return result

    def _forward_inplace(self, x: Tensor) -> Tensor:
        # Zero out negative values
        pos_mask = x >= 0
        x[~pos_mask] = 0

        if pos_mask.any():
            if self.training:
                # Sample k_i from U(1-alpha, 1+alpha) during training
                k = torch.empty_like(x).uniform_(1 - self.alpha, 1 + self.alpha)
                x[pos_mask] *= k[pos_mask]
            # For testing, we keep x[pos_mask] unchanged since k_i = 1

        return x


@register_activation
class AppReLU(BaseActivation):
    r"""
    Applies the Approximated ReLU (AppReLU) activation function:

    .. math::
        \text{AppReLU}(z) =
        \begin{cases}
        a z^b, & z \geq 0, \\
        0, & z < 0,
        \end{cases}

    Args:
        a (float, optional): Scale parameter. Default: ``1.0``
        b (float, optional): Power parameter. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/AppReLU.png

    Examples::

        >>> m = torch_activation.AppReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AppReLU(a=0.5, b=2.0)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, self.a * x.clamp(min=0) ** self.b, torch.zeros_like(x))


@register_activation
class ABReLU(BaseActivation):
    r"""
    Applies the Adaptive Bilateral ReLU (ABReLU) activation function:

    :math:`\text{ABReLU}(z_i) = \max(0, z_i - \bar{z})`

    where :math:`\bar{z}` is the mean of the input tensor.

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ABReLU.png

    Examples::

        >>> m = torch_activation.ABReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ABReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return F.relu(x - x.mean())


@register_activation
class DelayReLU(BaseActivation):
    r"""
    Applies the Delayed ReLU activation function:

    :math:`\text{DelayReLU}(z) = \max(0, z - a)`

    Args:
        a (float, optional): Delay threshold. Default: ``0.5``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DelayReLU.png

    Examples::

        >>> m = torch_activation.DelayReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DelayReLU(a=1.0)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return F.relu(x - self.a)


@register_activation
class DisReLU(BaseActivation):
    r"""
    Applies the Displaced ReLU (DisReLU) activation function:

    .. math::
        \text{DisReLU}(z) =
        \begin{cases}
        z, & z + a \geq 0, \\
        -a, & z + a < 0,
        \end{cases}

    Args:
        a (float, optional): Displacement parameter. Default: ``0.5``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DisReLU.png

    Examples::

        >>> m = torch_activation.DisReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.DisReLU(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x + self.a >= 0, x, torch.full_like(x, -self.a))


@register_activation
class ModifiedLReLU(BaseActivation):
    r"""
    Applies the Modified Leaky ReLU activation function:

    .. math::
        \text{ModifiedLReLU}(z) =
        \begin{cases}
        z, & z + a > 0, \\
        -az, & z + a \leq 0,
        \end{cases}

    Args:
        a (float, optional): Leakage parameter. Default: ``0.1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ModifiedLReLU.png

    Examples::

        >>> m = torch_activation.ModifiedLReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ModifiedLReLU(a=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x + self.a > 0, x, -self.a * x)


@register_activation
class FlattedTSwish(BaseActivation):
    r"""
    Applies the Flatted-T Swish activation function:

    .. math::
        \text{FlattedTSwish}(z) =
        \begin{cases}
        z \cdot \sigma(z) + T, & z \geq 0, \\
        T, & z < 0,
        \end{cases}

    where :math:`\sigma` is the sigmoid function and :math:`T = -0.20`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/FlattedTSwish.png

    Examples::

        >>> m = torch_activation.FlattedTSwish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.FlattedTSwish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    T: float = -0.20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x * torch.sigmoid(x) + self.T, torch.full_like(x, self.T))


@register_activation
class OAF(BaseActivation):
    r"""
    Applies the Output Activation Function (OAF):

    .. math::
        \text{OAF}(z) =
        \begin{cases}
        z + z \cdot \sigma(z), & z \geq 0, \\
        z \cdot \sigma(z), & z < 0,
        \end{cases}

    where :math:`\sigma` is the sigmoid function.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/OAF.png

    Examples::

        >>> m = torch_activation.OAF()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.OAF()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        sw = x * torch.sigmoid(x)
        return torch.where(x >= 0, x + sw, sw)


@register_activation
class SurveyELU(BaseActivation):
    r"""
    Applies the survey variant of ELU where the alpha scales the denominator:

    .. math::
        \text{SurveyELU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        \frac{\exp(z) - 1}{a}, & z < 0,
        \end{cases}

    Args:
        a (float, optional): Denominator scale for negative part. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SurveyELU.png

    Examples::

        >>> m = torch_activation.SurveyELU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SurveyELU(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, (torch.exp(x) - 1) / self.a)


@register_activation
class REU(BaseActivation):
    r"""
    Applies the Rectified Exponential Unit (REU) activation function:

    .. math::
        \text{REU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        z \cdot \exp(z), & z < 0,
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/REU.png

    Examples::

        >>> m = torch_activation.REU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.REU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, x * torch.exp(x))


@register_activation
class ADA(BaseActivation):
    r"""
    Applies the ADA activation function:

    .. math::
        \text{ADA}(z) =
        \begin{cases}
        \exp(-az + b), & z \geq 0, \\
        0, & z < 0,
        \end{cases}

    Args:
        a (float, optional): Decay rate for positive part. Default: ``1.0``
        b (float, optional): Offset for positive part. Default: ``0.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ADA.png

    Examples::

        >>> m = torch_activation.ADA()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ADA(a=0.5, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, torch.exp(-self.a * x + self.b), torch.zeros_like(x))


@register_activation
class LADA(BaseActivation):
    r"""
    Applies the LADA activation function:

    .. math::
        \text{LADA}(z) =
        \begin{cases}
        \exp(-az + b), & z \geq 0, \\
        cz, & z < 0,
        \end{cases}

    Args:
        a (float, optional): Decay rate for positive part. Default: ``1.0``
        b (float, optional): Offset for positive part. Default: ``0.0``
        c (float, optional): Slope for negative part. Default: ``0.1``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LADA.png

    Examples::

        >>> m = torch_activation.LADA()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LADA(a=0.5, b=1.0, c=0.2)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, torch.exp(-self.a * x + self.b), self.c * x)


@register_activation
class SigLU(BaseActivation):
    r"""
    Applies the Sigmoid Linear Unit variant (SigLU):

    .. math::
        \text{SigLU}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        \tanh(z), & z < 0,
        \end{cases}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SigLU.png

    Examples::

        >>> m = torch_activation.SigLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SigLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, torch.tanh(x))


@register_activation
class SaRa(BaseActivation):
    r"""
    Applies the SaRa activation function:

    .. math::
        \text{SaRa}(z) =
        \begin{cases}
        z, & z \geq 0, \\
        \dfrac{z}{1 + a \exp(-bz)}, & z < 0,
        \end{cases}

    Args:
        a (float, optional): Scale in denominator. Default: ``1.0``
        b (float, optional): Rate in denominator. Default: ``1.0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SaRa.png

    Examples::

        >>> m = torch_activation.SaRa()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SaRa(a=2.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, x / (1 + self.a * torch.exp(-self.b * x)))


@register_activation
class ShiftedReLU(BaseActivation):
    r"""
    Applies the Shifted ReLU activation function:

    .. math::
        \text{ShiftedReLU}(z) = \max(0, z + a)

    Args:
        a (float, optional): Shift parameter. Default: ``-0.5``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ShiftedReLU.png

    Examples::

        >>> m = torch_activation.ShiftedReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ShiftedReLU(a=-1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = -0.5, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.a)


@register_activation
class AllReLU(BaseActivation):
    r"""
    Applies the All-ReLU activation function, which applies a scaled ReLU on the negative part
    with sign depending on layer parity:

    .. math::
        \text{AllReLU}(z_i) =
        \begin{cases}
        -a z_i, & z_i \leq 0 \text{ and } l \text{ even}, \\
        a z_i, & z_i \leq 0 \text{ and } l \text{ odd}, \\
        z_i, & z_i > 0,
        \end{cases}

    Args:
        a (float, optional): Scale for negative part. Default: ``0.1``
        layer (int, optional): Layer index (parity determines sign of negative response). Default: ``0``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/AllReLU.png

    Examples::

        >>> m = torch_activation.AllReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AllReLU(a=0.2, layer=1)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.1, layer: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.layer = layer

    def _forward(self, x: Tensor) -> Tensor:
        neg_scale = -self.a if self.layer % 2 == 0 else self.a
        return torch.where(x > 0, x, neg_scale * x)


if __name__ == "__main__":
    from torch_activation.utils import plot_activation

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    activation_params = {
        "ReLU": {},
        "SReLU": {},
        # "SoftsignRReLU": {"l": [1/8, 1/4], "u": [1/5, 1/2]},
        "SlReLU": {},
        "CReLU": {},
        "ReLUN": {"n": [1, 6]},
        "SquaredReLU": {},
        "SineReLU": {"a": [0.5, 2]},
        "Minsin": {},
        "VLU": {"a": [0.5, 2], "b": [0.5, 2]},
        "LReLU": {"a": [50, 100]},
        "RReLU": {"l": [2, 3], "u": [6, 8]},
        # "SRReLU": {"l": [1/8, 1/4], "u": [1/5, 1/2]},
        "NReLU": {},
        "RTReLU": {"sigma": [0.5, 1.0]},
        "NLReLU": {"a": [0.5, 1, 2]},
        "SLU": {},
        "ReSP": {"a": [1.5, 2.0]},
        "PReNU": {"a": [0.1, 0.5]},
        "BReLU": {"a": [1, 3, 6]},
        "HardSigmoid": {"version": [1, 2]},
        "HardTanh": {"a": [-2, -1], "b": [1, 2]},
        "SvHardTanh": {"a": [0, 0.5, 1]},
        "ShHardTanh": {"a": [0, 0.5, 1]},
        "HardSwish": {},
        "TRec": {"a": [0.5, 1, 2]},
        "Hardshrink": {"a": [0.5, 1, 2]},
        "Softshrink": {"a": [0.5, 1, 2]},
        "BLReLU": {"a": [0.1, 0.2], "b": [1, 2]},
        "VReLU": {},
        "PanFunction": {"a": [0.5, 2]},
        "AbsLU": {"a": [0.2, 0.8]},
        "mReLU": {},
        "LSPTLU": {"a": [0.5, 1, 2]},
        "SoftModulusT": {},
        "SoftModulusQ": {},
        "SignReLU": {"a": [0.5, 1, 2]},
        "LiReLU": {},
        # "SCAA": {}  # NOTE: SCAA is not one-to-one.
    }

    for activation_name, params in activation_params.items():
        # Get the class from its name
        activation_class = globals()[activation_name]
        plot_activation(activation_class, params)
