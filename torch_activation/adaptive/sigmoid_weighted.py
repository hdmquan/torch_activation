import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class Swish(BaseActivation):
    r"""
    Applies the Swish activation function:

    :math:`\text{Swish}(x) = x \cdot \sigma(a \cdot x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Swish.png

    Examples::

        >>> m = torch_activation.Swish(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Swish(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, learnable: bool = False, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def extra_repr(self):
        a_val = self.a.item() if hasattr(self.a, "item") else self.a
        return f"a={a_val:.4f}"

    def _forward(self, x: Tensor) -> Tensor:
        result = x * torch.sigmoid(self.a * x)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class AHAF(BaseActivation):
    r"""
    Applies the Adaptive Hybrid Activation Function:

    :math:`\text{AHAF}(x) = a \cdot x \cdot \sigma(b \cdot x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Scaling parameter. Default: ``1.0``
        b (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/AHAF.png

    Examples::

        >>> m = torch_activation.AHAF(a=1.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AHAF(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x: Tensor) -> Tensor:
        result = self.a * x * torch.sigmoid(self.b * x)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSSiLU(BaseActivation):
    r"""
    Applies the Parametric Shifted SiLU activation function:

    :math:`\text{PSSiLU}(x) = x \cdot \frac{\sigma(a \cdot x) - b}{1 - b}`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        b (float, optional): Shift parameter. Default: ``0.5``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PSSiLU.png

    Examples::

        >>> m = torch_activation.PSSiLU(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSSiLU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.5,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Ensure b is less than 1 to avoid division by zero
            self.b = nn.Parameter(Tensor([min(b, 0.99)]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([min(b, 0.99)])  # Ensure b is less than 1

    def _forward(self, x: Tensor) -> Tensor:
        # Compute the shifted and normalized sigmoid
        shifted_sigmoid = (torch.sigmoid(self.a * x) - self.b) / (1 - self.b)
        result = x * shifted_sigmoid

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ESwish(BaseActivation):
    r"""
    Applies the E-Swish activation function:

    :math:`\text{E-swish}(x) = a \cdot x \cdot \sigma(x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Scaling parameter, recommended in range [1, 2]. Default: ``1.5``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ESwish.png

    Examples::

        >>> m = torch_activation.ESwish(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ESwish(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.5, learnable: bool = False, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x: Tensor) -> Tensor:
        result = self.a * x * torch.sigmoid(x)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ACONB(BaseActivation):
    r"""
    Applies the ACON-B activation function:

    :math:`\text{ACON-B}(x) = (1 - b) \cdot x \cdot \sigma(a \cdot (1 - b) \cdot x) + b \cdot x`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        b (float, optional): Parameter controlling the linear component. Default: ``0.25``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ACONB.png

    Examples::

        >>> m = torch_activation.ACONB(a=1.0, b=0.25)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ACONB(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.25,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Ensure b is between 0 and 1
            self.b = nn.Parameter(Tensor([max(0.0, min(b, 1.0))]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([max(0.0, min(b, 1.0))])  # Ensure b is between 0 and 1

    def _forward(self, x: Tensor) -> Tensor:
        one_minus_b = 1 - self.b
        swish_part = one_minus_b * x * torch.sigmoid(self.a * one_minus_b * x)
        linear_part = self.b * x
        result = swish_part + linear_part

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ACONC(BaseActivation):
    r"""
    Applies the ACON-C activation function:

    :math:`\text{ACON-C}(x) = (c - b) \cdot x \cdot \sigma(a \cdot (c - b) \cdot x) + b \cdot x`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        b (float, optional): Parameter controlling the linear component. Default: ``0.0``
        c (float, optional): Parameter controlling the swish component. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ACONC.png

    Examples::

        >>> m = torch_activation.ACONC(a=1.0, b=0.0, c=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ACONC(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 1.0,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def _forward(self, x: Tensor) -> Tensor:
        c_minus_b = self.c - self.b
        swish_part = c_minus_b * x * torch.sigmoid(self.a * c_minus_b * x)
        linear_part = self.b * x
        result = swish_part + linear_part

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSGU(BaseActivation):
    r"""
    Applies the Parameterized Self-Circulating Gating Unit activation function:

    :math:`\text{PSGU}(x) = x \cdot \tanh(a \cdot \sigma(x))`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``0.5``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PSGU.png

    Examples::

        >>> m = torch_activation.PSGU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSGU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.5, learnable: bool = False, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x: Tensor) -> Tensor:
        result = x * torch.tanh(self.a * torch.sigmoid(x))

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TBSReLUl(BaseActivation):
    r"""
    Applies the Tangent-Bipolar-Sigmoid ReLU Learnable activation function:

    :math:`\text{TBSReLUl}(x) = x \cdot \tanh\left(a \cdot \frac{1 - \exp(-x)}{1 + \exp(-x)}\right)`

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``0.5``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TBSReLUl.png

    Examples::

        >>> m = torch_activation.TBSReLUl(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TBSReLUl(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.5, learnable: bool = False, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x: Tensor) -> Tensor:
        exp_neg = torch.exp((-x).clamp(max=88.0))
        bipolar_sigmoid = (1 - exp_neg) / (1 + exp_neg)
        result = x * torch.tanh(self.a * bipolar_sigmoid)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PATS(BaseActivation):
    r"""
    Applies the PATS activation function:

    :math:`\text{PATS}(x) = x \cdot \arctan(a \cdot \pi \cdot \sigma(x))`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``0.625``
        lower_bound (float, optional): Lower bound for sampling a. Default: ``0.5``
        upper_bound (float, optional): Upper bound for sampling a. Default: ``0.75``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PATS.png

    Examples::

        >>> m = torch_activation.PATS(a=0.625)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PATS(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 0.625,
        lower_bound: float = 0.5,
        upper_bound: float = 0.75,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if learnable:
            # Initialize with a value in the valid range
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x: Tensor) -> Tensor:
        # If not in training mode or not learnable, use the fixed parameter
        if not self.training or not isinstance(self.a, nn.Parameter):
            a_value = self.a
        else:
            # During training with learnable parameter, sample from uniform distribution
            a_value = (
                torch.rand_like(self.a) * (self.upper_bound - self.lower_bound) + self.lower_bound
            )

        result = x * torch.arctan(a_value * math.pi * torch.sigmoid(x))

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class AQuLU(BaseActivation):
    r"""
    Applies the Adaptive Quadratic Linear Unit activation function:

    .. math::

        \text{AQuLU}(x) = \begin{cases}
            x, & x \geq \frac{1 - b}{a} \\
            a \cdot x^2 + b \cdot x, & -\frac{b}{a} \leq x < \frac{1 - b}{a} \\
            0, & x < -\frac{b}{a}
        \end{cases}

    Args:
        a (float, optional): Parameter controlling the quadratic component. Default: ``0.2``
        b (float, optional): Parameter controlling the linear component. Default: ``0.1``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/AQuLU.png

    Examples::

        >>> m = torch_activation.AQuLU(a=0.2, b=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AQuLU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.1,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            # Ensure a is positive to avoid division by zero
            self.a = nn.Parameter(Tensor([max(1e-6, a)]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([max(1e-6, a)])
            self.b = Tensor([b])

    def _forward(self, x: Tensor) -> Tensor:
        a = self.a.to(x.dtype) if isinstance(self.a, Tensor) else x.new_tensor(self.a)
        b = self.b.to(x.dtype) if isinstance(self.b, Tensor) else x.new_tensor(self.b)
        upper_threshold = (1 - b) / a
        lower_threshold = -b / a

        mask_upper = x >= upper_threshold
        mask_middle = (x >= lower_threshold) & (x < upper_threshold)
        mask_lower = x < lower_threshold

        if self.inplace:
            result = x.clone()

            result[mask_upper] = x[mask_upper]
            result[mask_middle] = a * x[mask_middle] ** 2 + b * x[mask_middle]
            result[mask_lower] = 0

            x.copy_(result)
            return x
        else:
            result = torch.zeros_like(x)

            result[mask_upper] = x[mask_upper]
            result[mask_middle] = a * x[mask_middle] ** 2 + b * x[mask_middle]

            return result


@register_activation
class SinLU(BaseActivation):
    r"""
    Applies the Sinu-Sigmoidal Linear Unit activation function:

    :math:`\text{SinLU}(x) = (x + a \cdot \sin(b \cdot x)) \cdot \sigma(x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Amplitude parameter for sine component. Default: ``0.5``
        b (float, optional): Frequency parameter for sine component. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SinLU.png

    Examples::

        >>> m = torch_activation.SinLU(a=0.5, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SinLU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 0.5,
        b: float = 1.0,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x: Tensor) -> Tensor:
        modified_x = x + self.a * torch.sin(self.b * x)
        result = modified_x * torch.sigmoid(x)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ErfAct(BaseActivation):
    r"""
    Applies the ErfAct activation function:

    :math:`\text{ErfAct}(x) = x \cdot \text{erf}(a \cdot \exp(b \cdot x))`

    where :math:`\text{erf}(x)` is the error function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        b (float, optional): Parameter controlling the exponential growth. Default: ``0.5``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ErfAct.png

    Examples::

        >>> m = torch_activation.ErfAct(a=1.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ErfAct(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.5,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x: Tensor) -> Tensor:
        # Calculate exp(b*x) with clipping to prevent overflow
        exp_term = torch.exp(torch.clamp(self.b * x, max=20))
        result = x * torch.erf(self.a * exp_term)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSerf(BaseActivation):
    r"""
    Applies the Parametric Serf activation function:

    :math:`\text{pserf}(x) = x \cdot \text{erf}(a \cdot \ln(1 + \exp(b \cdot x)))`

    where :math:`\text{erf}(x)` is the error function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``1.0``
        b (float, optional): Parameter controlling the softplus term. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PSerf.png

    Examples::

        >>> m = torch_activation.PSerf(a=1.0, b=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSerf(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        learnable: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x: Tensor) -> Tensor:
        # Calculate softplus: ln(1 + exp(b*x))
        softplus = torch.log1p(torch.exp(torch.clamp(self.b * x, max=20)))
        result = x * torch.erf(self.a * softplus)

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class Swim(BaseActivation):
    r"""
    Applies the Swim activation function:

    :math:`\text{Swim}(x) = x \cdot \frac{1}{2} \left(1 + \frac{a \cdot x}{\sqrt{1 + x^2}}\right)`

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: ``0.5``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Swim.png

    Examples::

        >>> m = torch_activation.Swim(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Swim(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.5, learnable: bool = False, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x: Tensor) -> Tensor:
        # Calculate the modified sigmoid-like term
        sigmoid_term = 0.5 * (1 + (self.a * x) / torch.sqrt(1 + x.pow(2)))
        result = x * sigmoid_term

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class GPSoftmax(BaseActivation):
    r"""
    Applies the Generalized Power Softmax activation function:

    :math:`\text{GPSoftmax}(z_j) = \frac{\exp(\text{PNORM}(z_j))}{\sum_{k=1}^{N} \exp(\text{PNORM}(z_k))}`

    where :math:`\text{PNORM}(z_i) = \frac{z_i - M_{a_i, b_i}}{\text{GPM}_{c_i, d_i}(z - M_{a_i, b_i})}`, :math:`M_{a_i, b_i} = \text{GPM}_{a_i, b_i}(z)`, and :math:`\text{GPM}_{\alpha, \beta}(x) = \frac{\ln\left(\sum_{k=1}^{N} \alpha^{\beta x_k}\right) - \ln(N)}{\beta \ln(\alpha)}`.

    Args:
        input_shape (int): The size of the input vector tensor, channel or feature size.
        a (float, optional): Initial value for parameter a. Default: ``1.0``
        b (float, optional): Initial value for parameter b. Default: ``1.0``
        c (float, optional): Initial value for parameter c. Default: ``1.0``
        d (float, optional): Initial value for parameter d. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GPSoftmax.png

    Examples::

        >>> m = torch_activation.GPSoftmax(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GPSoftmax(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            """Creates a learnable parameter if `learnable` is True; otherwise, returns a fixed tensor."""  # noqa: E501
            tensor = torch.full(
                (input_shape, 1), value, dtype=torch.float64
            )  # Initialize tensor with the given value
            return (
                nn.Parameter(torch.randn(input_shape)) if learnable else tensor
            )  # Convert to parameter if learnable

        # Initialize parameters (either as learnable or fixed tensors)
        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.c: Tensor = create_param(c)
        self.d: Tensor = create_param(d)
        self.inplace: bool = inplace
        self.input_size = input_shape

    def _forward(self, x: Tensor) -> Tensor:
        """
        Computes the generalized Lehmer softmax transformation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Softmax-transformed tensor.
        """
        result = F.softmax(self.pnorm(x, self.a, self.b, self.c, self.d), dim=-1)
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result

    def pnorm(self, x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        """
        Applies Lehmer-based normalization:

        PNORM(z_i) = (z_i - M_{a_i, b_i}) / GPM_{c_i, d_i}(z - M_{a_i, b_i})
        """
        glm_first: Tensor = self.gpm_func(x, a, b)
        glm_second: Tensor = self.gpm_func(x - glm_first, c, d)
        result: Tensor = (x - glm_first) / glm_second

        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result

    def gpm_func(self, x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """
        Computes the Generalized Power Mean (GPM):

        GPM_{\alpha, \beta}(x) = (ln( sum(\alpha^{\beta x_k}) ) - ln(N)) / (\beta ln(\alpha))
        """
        log_alpha: Tensor = torch.log(torch.clamp(alpha, min=1e-8))
        b: Tensor = torch.multiply(beta, x)
        first_part: Tensor = torch.logsumexp(b * log_alpha, dim=-1, keepdim=True)

        second_part = torch.log(torch.tensor(float(self.input_size)))

        denom_part = torch.multiply(beta, log_alpha)
        res: Tensor = (first_part - second_part) / denom_part
        return res


@register_activation
class GLSoftmax(BaseActivation):
    r"""
    Applies the Generalized Lehmer Softmax activation function:

    :math:`\text{GLSoftmax}(z_j) = \frac{\exp(\text{LNORM}(z_j))}{\sum_{k=1}^{N} \exp(\text{LNORM}(z_k))}`

    where :math:`\text{LNORM}(z_i) = \frac{z_i - M_{a_i, b_i}}{\text{GLM}_{c_i, d_i}(z - M_{a_i, b_i})}`, :math:`M_{a_i, b_i} = \text{GLM}_{a_i, b_i}(z)`, and :math:`\text{GLM}_{\alpha, \beta}(x) = \frac{\ln \left( \frac{\sum_{k=1}^{N} \alpha^{(\beta+1)x_k}}{\sum_{k=1}^{N} \alpha^{\beta x_k}} \right)}{\ln(\alpha)}`.

    Args:
        input_shape (int): The size of the input vector tensor, channel or feature size.
        a (float, optional): Initial value for parameter a. Default: ``1.0``
        b (float, optional): Initial value for parameter b. Default: ``1.0``
        c (float, optional): Initial value for parameter c. Default: ``1.0``
        d (float, optional): Initial value for parameter d. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GLSoftmax.png

    Examples::

        >>> m = torch_activation.GLSoftmax(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GLSoftmax(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            """Creates a learnable parameter if `learnable` is True; otherwise, returns a fixed tensor."""  # noqa: E501
            tensor = torch.full(
                (input_shape, 1), value, dtype=torch.float64
            )  # Initialize tensor with the given value
            return (
                nn.Parameter(torch.randn(input_shape)) if learnable else tensor
            )  # Convert to parameter if learnable

        # Initialize parameters (either as learnable or fixed tensors)
        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.c: Tensor = create_param(c)
        self.d: Tensor = create_param(d)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        """
        Computes the generalized Lehmer softmax transformation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Softmax-transformed tensor.
        """
        result = F.softmax(self.lnorm(x, self.a, self.b, self.c, self.d), dim=-1)
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result

    def lnorm(self, x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        """
        Applies Lehmer-based normalization.

        Args:
            x (Tensor): Input tensor.
            a (Tensor): Parameter `a`.
            b (Tensor): Parameter `b`.
            c (Tensor): Parameter `c`.
            d (Tensor): Parameter `d`.

        Returns:
            Tensor: Normalized tensor.
        """
        glm_first: Tensor = self.glm_func(x, a, b)
        glm_second: Tensor = self.glm_func(x - glm_first, c, d)
        result: Tensor = (x - glm_first) / glm_second
        return result

    def glm_func(self, x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """
        Computes the generalized Lehmer mean function.

        Args:
            x (Tensor): Input tensor.
            alpha (Tensor): Alpha parameter.
            beta (Tensor): Beta parameter.

        Returns:
            Tensor: Result of the generalized Lehmer mean function.
        """
        log_alpha: Tensor = torch.log(torch.clamp(alpha, min=1e-8))
        b: Tensor = torch.multiply(beta + 1, x)
        first_part: Tensor = torch.logsumexp(b * log_alpha, dim=-1, keepdim=True)
        b = torch.multiply(beta, x)
        second_part: Tensor = torch.logsumexp(b * log_alpha, dim=-1, keepdim=True)
        res: Tensor = (first_part - second_part) / log_alpha
        return res


@register_activation
class ARBF(BaseActivation):
    r"""
    Applies the Adaptive Radial Basis Function activation function:

    :math:`\text{ARBF}(z_i) = \exp \left( -\frac{(z_i - a_i)^2}{2b_i^2} \right)`

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for the center parameter. Default: ``1.0``
        b (float, optional): Initial value for the width parameter. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ARBF.png

    Examples::

        >>> m = torch_activation.ARBF(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ARBF(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 1.0,
        b: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = torch.exp(-0.5 * (x - self.a) ** 2 / torch.pow(self.b, 2))
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class PGELU(BaseActivation):
    r"""
    Applies the Parametric Gaussian Error Linear Unit activation function:

    :math:`\text{PGELU}(z_i) = z \cdot \Phi \left( \frac{z}{a} \right)`

    where :math:`\Phi(z)` is the standard Gaussian cumulative distribution function.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for the RMS noise parameter. Default: ``1.0``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PGELU.png

    Examples::

        >>> m = torch_activation.PGELU(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PGELU(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = x * 0.5 * (1 + torch.erf((x / self.a) / math.sqrt(2)))
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class PFTS(BaseActivation):
    r"""
    Applies the Parametric Flatted-T Swish activation function:

    .. math::

        \text{PFTS}(z_i) = \begin{cases}
            \frac{z_i}{1+\exp(-z_i)} + T_i, & z_i \geq 0 \\
            T_i, & z_i < 0
        \end{cases}

    where :math:`T_i` is a trainable parameter.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        T (float, optional): Initial value for the trainable parameter T. Default: ``-0.2``
        learnable (bool, optional): optionally make ``T`` trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PFTS.png

    Examples::

        >>> m = torch_activation.PFTS(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PFTS(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        T: float = -0.2,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(-0.2 * torch.ones(input_shape)) if learnable else tensor

        self.T: Tensor = create_param(T)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = torch.nn.functional.relu(x) * torch.nn.functional.sigmoid(x) + self.T
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class PFPM(BaseActivation):
    r"""
    Applies the Parametric Flatten-p Mish activation function:

    .. math::

        \text{PFPM}(z_i) = \begin{cases}
            z_i \tanh(\ln(1 + \exp(z_i))) + p_i, & z_i \geq 0 \\
            p_i, & z_i < 0
        \end{cases}

    where :math:`p_i` is a trainable parameter.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        p (float, optional): Initial value for the trainable parameter p. Default: ``1.0``
        learnable (bool, optional): optionally make ``p`` trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PFPM.png

    Examples::

        >>> m = torch_activation.PFPM(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PFPM(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        p: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.p: Tensor = create_param(p)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = torch.nn.functional.relu(x) * torch.tanh(torch.log1p(torch.exp(x))) + self.p
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class PSIGRAMP(BaseActivation):
    r"""
    Applies the Parametric Sigmoid-Ramp activation function:

    .. math::

        \text{PSIGRAMP}(z_i) = a_i \sigma(z_i) + (1 - a_i) \cdot \begin{cases}
            1, & z_i \geq \frac{1}{2b_i} \\
            b_i z_i + \frac{1}{2}, & -\frac{1}{2b_i} < z_i < \frac{1}{2b_i} \\
            0, & z_i \leq -\frac{1}{2b_i}
        \end{cases}

    where :math:`a_i` is constrained to :math:`[0, 1]`.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for the sigmoid blend parameter. Default: ``0.5``
        b (float, optional): Initial value for the ramp slope parameter. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PSIGRAMP.png

    Examples::

        >>> m = torch_activation.PSIGRAMP(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSIGRAMP(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 0.5,
        b: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        converted_a = torch.nn.functional.sigmoid(self.a)
        relu_section = 0.5 * (
            torch.nn.functional.relu(2 * self.b * x + 1)
            - torch.nn.functional.relu(2 * self.b * x - 1)
        )

        result = converted_a * (torch.nn.functional.sigmoid(x)) + (1 - converted_a) * relu_section
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class RSIGN(BaseActivation):
    r"""
    Applies the React-Sign activation function:

    .. math::

        \text{RSIGN}(z_i) = \begin{cases}
            1, & z_i \geq a_c \\
            -1, & z_i < a_c
        \end{cases}

    where :math:`a_c` is an adaptive threshold parameter for each channel.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for the adaptive threshold. Default: ``0.5``
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/RSIGN.png

    Examples::

        >>> m = torch_activation.RSIGN(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.RSIGN(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 0.5,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        func1 = torch.sign(x - self.a)
        result = torch.where(func1 == 0, torch.tensor(1), func1)
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class MAF(BaseActivation):
    r"""
    Applies the Multiquadratic Activation Function:

    :math:`\text{MAF}(z_i) = \sqrt{\|z_i - a_i\|^2 + b_i^2}`

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for the slope coefficient. Default: ``0.5``
        b (float, optional): Initial value for the bias coefficient. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MAF.png

    Examples::

        >>> m = torch_activation.MAF(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MAF(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 0.5,
        b: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = torch.sqrt((x - self.a) ** 2 + torch.pow(self.b, 2))
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class UAF(BaseActivation):
    r"""
    Applies the Universal Activation Function:

    :math:`\text{UAF}(z_i) = \ln(1 + \exp(a_i(z_i + b_i) + c_i z_i^2)) - \ln(1 + \exp(d_i(z_i - b_i))) + e_i`

    Args:
        input_shape (int): Size of the input tensor (feature size).
        a (float, optional): Initial value for parameter a. Default: ``0.5``
        b (float, optional): Initial value for parameter b. Default: ``1.0``
        c (float, optional): Initial value for parameter c. Default: ``0.5``
        d (float, optional): Initial value for parameter d. Default: ``1.0``
        e (float, optional): Initial value for parameter e. Default: ``0.5``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/UAF.png

    Examples::

        >>> m = torch_activation.UAF(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.UAF(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 0.5,
        d: float = 1.0,
        e: float = 0.5,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape,), value)
            return nn.Parameter(tensor) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.c: Tensor = create_param(c)
        self.d: Tensor = create_param(d)
        self.e: Tensor = create_param(e)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        term1 = F.softplus(self.a * (x + self.b) + self.c * x**2)
        term2 = F.softplus(self.d * (x - self.b))
        result = term1 - term2 + self.e
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class GReLU(BaseActivation):
    r"""
    Applies the Generalized Rectified Linear Unit activation function:

    :math:`\text{GReLU}(z_i) = \frac{\ln(1 + a_i^{b_i z_i})}{b_i \ln(a_i)}`

    where :math:`a_i > 1` and :math:`b_i > 0` are trainable parameters constrained via softplus.

    Args:
        input_shape (int): Size of the input tensor (feature size).
        a (float, optional): Initial value for parameter a. Default: ``1.5``
        b (float, optional): Initial value for parameter b. Default: ``0.5``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GReLU.png

    Examples::

        >>> m = torch_activation.GReLU(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GReLU(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 1.5,
        b: float = 0.5,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        a = 1 + F.softplus(self.a)
        b = F.softplus(self.b)
        term1 = F.softplus(torch.log(a) * b * x)
        term2 = b * torch.log(a)
        result = term1 / term2
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result


@register_activation
class GLN(BaseActivation):
    r"""
    Applies the Global-Local Neuron activation function:

    :math:`\text{GLN}(z_l) = \sigma(a_l) \cdot \sin(z_l) + (1 - \sigma(a_l)) \cdot \tanh(z_l) - b_l`

    where :math:`\sigma(a_l)` is the sigmoid gate blending global (sin) and local (tanh) activations.

    Args:
        input_shape (int): Size of the input tensor (feature size).
        a (float, optional): Initial value for the gating parameter. Default: ``1.0``
        b (float, optional): Initial value for the bias parameter. Default: ``1.0``
        learnable (bool, optional): optionally make parameters trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/GLN.png

    Examples::

        >>> m = torch_activation.GLN(input_shape=4)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GLN(input_shape=8, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self,
        input_shape: int,
        a: float = 1.0,
        b: float = 1.0,
        learnable: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:

        result = F.sigmoid(self.a) * torch.sin(x) + (1 - F.sigmoid(self.a)) * torch.tanh(x) - self.b
        if self.inplace and hasattr(x, "copy_"):
            x.copy_(result)
            return x
        return result
