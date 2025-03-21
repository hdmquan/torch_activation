import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_activation.base import BaseActivation
from torch import Tensor
import math

from torch_activation import register_activation


@register_activation
class Swish(BaseActivation):
    r"""
    Applies the Swish activation function:

    :math:`\text{Swish}(x) = x \cdot \sigma(a \cdot x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.Swish(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Swish(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
            self,
            a: float = 1.0,
            learnable: bool = False,
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x) -> Tensor:
        result = x * torch.sigmoid(self.a * x)

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Scaling parameter. Default: 1.0
        b (float, optional): Parameter controlling the shape of the function. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x) -> Tensor:
        result = self.a * x * torch.sigmoid(self.b * x)

        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSSiLU(BaseActivation):
    r"""
    Applies the Parametric Shifted SiLU function:

    :math:`\text{PSSiLU}(x) = x \cdot \frac{\sigma(a \cdot x) - b}{1 - b}`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: 1.0
        b (float, optional): Shift parameter. Default: 0.5
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Ensure b is less than 1 to avoid division by zero
            self.b = nn.Parameter(Tensor([min(b, 0.99)]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([min(b, 0.99)])  # Ensure b is less than 1

    def _forward(self, x) -> Tensor:
        # Compute the shifted and normalized sigmoid
        shifted_sigmoid = (torch.sigmoid(self.a * x) - self.b) / (1 - self.b)
        result = x * shifted_sigmoid

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Scaling parameter, recommended in range [1, 2]. Default: 1.5
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ESwish(a=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ESwish(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
            self,
            a: float = 1.5,
            learnable: bool = False,
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x) -> Tensor:
        result = self.a * x * torch.sigmoid(x)

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Parameter controlling the shape of the function. Default: 1.0
        b (float, optional): Parameter controlling the linear component. Default: 0.25
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Ensure b is between 0 and 1
            self.b = nn.Parameter(Tensor([max(0.0, min(b, 1.0))]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([max(0.0, min(b, 1.0))])  # Ensure b is between 0 and 1

    def _forward(self, x) -> Tensor:
        one_minus_b = 1 - self.b
        swish_part = one_minus_b * x * torch.sigmoid(self.a * one_minus_b * x)
        linear_part = self.b * x
        result = swish_part + linear_part

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Parameter controlling the shape of the function. Default: 1.0
        b (float, optional): Parameter controlling the linear component. Default: 0.0
        c (float, optional): Parameter controlling the swish component. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def _forward(self, x) -> Tensor:
        c_minus_b = self.c - self.b
        swish_part = c_minus_b * x * torch.sigmoid(self.a * c_minus_b * x)
        linear_part = self.b * x
        result = swish_part + linear_part

        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSGU(BaseActivation):
    r"""
    Applies the Parameterized Self-Circulating Gating Unit function:

    :math:`\text{PSGU}(x) = x \cdot \tanh(a \cdot \sigma(x))`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: 0.5
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PSGU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSGU(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
            self,
            a: float = 0.5,
            learnable: bool = False,
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x) -> Tensor:
        result = x * torch.tanh(self.a * torch.sigmoid(x))

        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TBSReLUl(BaseActivation):
    r"""
    Applies the Tangent-Bipolar-Sigmoid ReLU Learnable function:

    :math:`\text{TBSReLUl}(x) = x \cdot \tanh\left(a \cdot \frac{1 - \exp(-x)}{1 + \exp(-x)}\right)`

    Args:
        a (float, optional): Parameter controlling the shape of the function. Default: 0.5
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TBSReLUl(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TBSReLUl(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
            self,
            a: float = 0.5,
            learnable: bool = False,
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x) -> Tensor:
        # Calculate bipolar sigmoid: (1 - exp(-x)) / (1 + exp(-x))
        bipolar_sigmoid = (1 - torch.exp(-x)) / (1 + torch.exp(-x))
        result = x * torch.tanh(self.a * bipolar_sigmoid)

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Parameter controlling the shape of the function. Default: 0.625
        lower_bound (float, optional): Lower bound for sampling a. Default: 0.5
        upper_bound (float, optional): Upper bound for sampling a. Default: 0.75
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if learnable:
            # Initialize with a value in the valid range
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x) -> Tensor:
        # If not in training mode or not learnable, use the fixed parameter
        if not self.training or not isinstance(self.a, nn.Parameter):
            a_value = self.a
        else:
            # During training with learnable parameter, sample from uniform distribution
            a_value = torch.rand_like(self.a) * (self.upper_bound - self.lower_bound) + self.lower_bound

        result = x * torch.arctan(a_value * math.pi * torch.sigmoid(x))

        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class AQuLU(BaseActivation):
    r"""
    Applies the Adaptive Quadratic Linear Unit function:

    :math:`\text{AQuLU}(x) = \begin{cases} 
        x, & x \geq \frac{1 - b}{a} \\
        a \cdot x^2 + b \cdot x, & -\frac{b}{a} \leq x < \frac{1 - b}{a} \\
        0, & x < -\frac{b}{a} 
    \end{cases}`

    Args:
        a (float, optional): Parameter controlling the quadratic component. Default: 0.2
        b (float, optional): Parameter controlling the linear component. Default: 0.1
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            # Ensure a is positive to avoid division by zero
            self.a = nn.Parameter(Tensor([max(1e-6, a)]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([max(1e-6, a)])
            self.b = Tensor([b])

    def _forward(self, x) -> Tensor:
        # Calculate thresholds
        upper_threshold = (1 - self.b) / self.a
        lower_threshold = -self.b / self.a

        # Create masks for different regions
        mask_upper = x >= upper_threshold
        mask_middle = (x >= lower_threshold) & (x < upper_threshold)
        mask_lower = x < lower_threshold

        if self.inplace:
            # Create a copy to avoid modifying during computation
            result = x.clone()

            # Apply different functions to different regions
            result[mask_upper] = x[mask_upper]
            result[mask_middle] = self.a * x[mask_middle] ** 2 + self.b * x[mask_middle]
            result[mask_lower] = 0

            # Copy back to original tensor
            x.copy_(result)
            return x
        else:
            # Initialize result tensor
            result = torch.zeros_like(x)

            # Apply different functions to different regions
            result[mask_upper] = x[mask_upper]
            result[mask_middle] = self.a * x[mask_middle] ** 2 + self.b * x[mask_middle]
            # result[mask_lower] is already 0

            return result


@register_activation
class SinLU(BaseActivation):
    r"""
    Applies the Sinu-Sigmoidal Linear Unit function:

    :math:`\text{SinLU}(x) = (x + a \cdot \sin(b \cdot x)) \cdot \sigma(x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Amplitude parameter for sine component. Default: 0.5
        b (float, optional): Frequency parameter for sine component. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x) -> Tensor:
        modified_x = x + self.a * torch.sin(self.b * x)
        result = modified_x * torch.sigmoid(x)

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Parameter controlling the shape of the function. Default: 1.0
        b (float, optional): Parameter controlling the exponential growth. Default: 0.5
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x) -> Tensor:
        # Calculate exp(b*x) with clipping to prevent overflow
        exp_term = torch.exp(torch.clamp(self.b * x, max=20))
        result = x * torch.erf(self.a * exp_term)

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Parameter controlling the shape of the function. Default: 1.0
        b (float, optional): Parameter controlling the softplus term. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

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
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def _forward(self, x) -> Tensor:
        # Calculate softplus: ln(1 + exp(b*x))
        softplus = torch.log(1 + torch.exp(torch.clamp(self.b * x, max=20)))
        result = x * torch.erf(self.a * softplus)

        if self.inplace and hasattr(x, 'copy_'):
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
        a (float, optional): Parameter controlling the shape of the function. Default: 0.5
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.Swim(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.Swim(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
            self,
            a: float = 0.5,
            learnable: bool = False,
            inplace: bool = False
            , **kwargs):
        super().__init__(**kwargs)

        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def _forward(self, x) -> Tensor:
        # Calculate the modified sigmoid-like term
        sigmoid_term = 0.5 * (1 + (self.a * x) / torch.sqrt(1 + x.pow(2)))
        result = x * sigmoid_term

        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class GPSoftmax(BaseActivation):
    """
    Generalized Power Softmax (gpsoftmax)

    This activation function extends the traditional softmax using a power-based normalization.
    It includes trainable parameters a, b, c, and d, which influence the transformation.

    :math:`f(z_j) = \frac{\exp(\text{PNORM}(z_j))}{\sum_{k=1}^{N} \exp(\text{PNORM}(z_k))}`

    where PNORM is a generalized power-based normalization:
    
    :math:`\text{PNORM}(z_i) = \frac{z_i - M_{a_i, b_i}}{\text{GPM}_{c_i, d_i}(z - M_{a_i, b_i})}`
    
    :math:`M_{a_i, b_i} = \text{GPM}_{a_i, b_i}(z)`
    
    :math:`\text{GPM}_{\alpha, \beta}(x) = \frac{\ln\left(\sum_{k=1}^{N} \alpha^{\beta x_k}\right) - \ln(N)}{\beta \ln(\alpha)}`

    
    Args:
        input_shape (int): The size of the input vector tensor, channel or feature size.
        a (float, optional): Initial value for parameter `a`. Default is 1.0.
        b (float, optional): Initial value for parameter `b`. Default is 1.0.
        c (float, optional): Initial value for parameter `c`. Default is 1.0.
        d (float, optional): Initial value for parameter `d`. Default is 1.0.
        learnable (bool, optional): Whether the parameters `a`, `b`, `c`, and `d` are trainable. Default is False.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for the `BaseActivation` superclass.

    Attributes:
        a (nn.Parameter or Tensor): Trainable parameter `a`.
        b (nn.Parameter or Tensor): Trainable parameter `b`.
        c (nn.Parameter or Tensor): Trainable parameter `c`.
        d (nn.Parameter or Tensor): Trainable parameter `d`.
        inplace (bool): If True, modifies the input tensor in place.
        input_shape (int): The size of the input vector tensor, channel or feature size.
    
    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the generalized Lehmer softmax transformation.

        pnorm(x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
            Applies Lehmer-based normalization.

        gpm_func(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
            Computes the generalized Lehmer mean function.
        
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
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            """Creates a learnable parameter if `learnable` is True; otherwise, returns a fixed tensor."""
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)  # Initialize tensor with the given value
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor  # Convert to parameter if learnable

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
        if self.inplace and hasattr(x, 'copy_'):
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

        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result

    def gpm_func(self, x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """
        Computes the Generalized Power Mean (GPM):

        GPM_{\alpha, \beta}(x) = (ln( sum(\alpha^{\beta x_k}) ) - ln(N)) / (\beta ln(\alpha))
        """
        b: Tensor = torch.multiply(beta + 1, x)
        log_alpha: Tensor = torch.log(torch.clamp(alpha, min=1e-8))
        first_part: Tensor = torch.logsumexp(b * log_alpha, dim=-1)

        second_part = torch.log(torch.Tensor([self.input_size]))

        denom_part = torch.multiply(beta, log_alpha)
        res: Tensor = (first_part - second_part) / denom_part
        return res


@register_activation
class GLSoftmax(BaseActivation):
    r"""
    Generalized Lehmer Softmax (glsoftmax).

    This is a softmax variant that applies a generalized Lehmer-based normalization
    with trainable parameters `a`, `b`, `c`, and `d`.
    
    :math:`f(z_j) = \frac{\exp(\text{LNORM}(z_j))}{\sum_{k=1}^{N} \exp(\text{LNORM}(z_k))}`

    where LNORM is a generalized Lehmer-based normalization:
    
    :math:`\text{LNORM}(z_i) = \frac{z_i - M_{a_i, b_i}}{\text{GLM}_{c_i, d_i}(z - M_{a_i, b_i})}`
    
    :math:`M_{a_i, b_i} = \text{GLM}_{a_i, b_i}(z)`
    
    :math:`\text{GLM}_{\alpha, \beta}(x) = \frac{\ln \left( \frac{\sum_{k=1}^{N} \alpha^{(\beta+1)x_k}}{\sum_{k=1}^{N} \alpha^{\beta x_k}} \right)}{\ln(\alpha)}`


    Args:
        input_shape (int): The size of the input vector tensor, channel or feature size.
        a (float, optional): Initial value for parameter `a`. Default is 1.0.
        b (float, optional): Initial value for parameter `b`. Default is 1.0.
        c (float, optional): Initial value for parameter `c`. Default is 1.0.
        d (float, optional): Initial value for parameter `d`. Default is 1.0.
        learnable (bool, optional): Whether the parameters `a`, `b`, `c`, and `d` are trainable. Default is False.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for the `BaseActivation` superclass.

    Attributes:
        a (Union[nn.Parameter, Tensor]): Trainable or fixed parameter `a`.
        b (Union[nn.Parameter, Tensor]): Trainable or fixed parameter `b`.
        c (Union[nn.Parameter, Tensor]): Trainable or fixed parameter `c`.
        d (Union[nn.Parameter, Tensor]): Trainable or fixed parameter `d`.
        inplace (bool): Flag indicating whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the generalized Lehmer softmax transformation.

        lnorm(x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
            Applies Lehmer-based normalization.

        glm_func(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
            Computes the generalized Lehmer mean function.
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
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            """Creates a learnable parameter if `learnable` is True; otherwise, returns a fixed tensor."""
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)  # Initialize tensor with the given value
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor  # Convert to parameter if learnable

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
        if self.inplace and hasattr(x, 'copy_'):
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
        b: Tensor = torch.multiply(beta + 1, x)
        log_alpha: Tensor = torch.log(torch.clamp(alpha, min=1e-8))
        first_part: Tensor = torch.logsumexp(b * log_alpha, dim=-1)
        b = torch.multiply(beta, x)
        log_alpha = torch.log(torch.clamp(alpha, min=1e-8))
        second_part: Tensor = torch.logsumexp(b * log_alpha, dim=-1)
        res: Tensor = (first_part - second_part) / torch.log(torch.clamp(alpha, min=1e-8))
        return res


@register_activation
class ARBF(BaseActivation):
    r"""

    Adaptive Radial Basis Function (ARBF) Model.

    This class implements an adaptive radial basis function as described in [499].
    The function is defined as:

    .. math:: ARBF(z_i) = \exp \left( -\frac{(z_i - a_i)^2}{2b_i^2} \right)

    where:
    - \( a_i \) is an adaptive parameter controlling the center of the neuron.
    - \( b_i \) is an adaptive parameter controlling the width* of the neuron.
    - \( z_i \) is the input variable.


    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for parameter `a`. Default is 1.0.
        b (float, optional): Initial value for parameter `b`. Default is 1.0.
        learnable (bool, optional): Whether `a` and `b` are trainable. Default is True.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for BaseActivation.

    Attributes:
        a (Tensor or nn.Parameter): Trainable or fixed parameter `a`.
        b (Tensor or nn.Parameter): Trainable or fixed parameter `b`.
        inplace (bool): Whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the Adaptive Radial Basis Function transformation.


    """

    def __init__(
            self,
            input_shape: int,
            a: float = 1.0,
            b: float = 1.0,
            learnable: bool = True,
            inplace: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = torch.exp(-0.5 * torch.nn.functional.mse_loss(x, self.a, reduction="none") / torch.pow(self.b, 2))
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result


@register_activation
class PGELU(BaseActivation):
    """
    Parametric Gaussian Error Linear Unit (PGELU).

    PGELU is an adaptive variant of GELU that incorporates noise injection.
    It is defined as:

    .. math:: PGELU(z_i) = z \cdot \Phi \left( \frac{z}{a} \right)

    where:
    - \( \Phi(z) \) is the standard Gaussian cumulative distribution function (CDF).
    - \( a \) is a learnable parameter representing root mean square (RMS) noise.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a (float, optional): Initial value for parameter `a`. Default is 1.0.
        learnable (bool, optional): Whether `a` is trainable. Default is True.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for BaseActivation.

    Attributes:
        a (Tensor or nn.Parameter): Trainable or fixed parameter `a`.
        inplace (bool): Whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the Parametric GELU transformation.

    """

    def __init__(
            self,
            input_shape: int,
            a: float = 1.0,
            b: float = 1.0,
            learnable: bool = True,
            inplace: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.a: Tensor = create_param(a)
        self.b: Tensor = create_param(b)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = x * 0.5 * (1 + torch.erf((x / self.a) / math.sqrt(2)))
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result


@register_activation
class PFTS(BaseActivation):
    """
    Parametric Flatted-T Swish (PFTS).

    PFTS is an adaptive extension of the Flatted-T Swish (FTS). It is identical to FTS except
    that the parameter T is adaptive.

    The PFTS activation function is defined as:

    .. math:: PFTS(z_i) = \text{ReLU}(z_i) \cdot \sigma(z_i) + T_i =
    \begin{cases}
    \frac{z_i}{1+\exp(-z_i)} + T_i, & z_i \geq 0, \\
    T_i, & z_i < 0,
    \end{cases}

    where:
    - ReLU(z_i) is the Rectified Linear Unit function applied to z_i.
    - σ(z_i) is the sigmoid function applied to z_i, i.e., 1 / (1 + exp(-z_i)).
    - T_i is a trainable parameter for each neuron i.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        T_i (float, optional): Initial value for parameter `T_i`. Default is -0.20.
        learnable (bool, optional): Whether `T_i` is trainable. Default is True.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for BaseActivation.

    Attributes:
        T_i (Tensor or nn.Parameter): Trainable or fixed parameter `T_i`.
        inplace (bool): Whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the Parametric Flatted-T Swish transformation.
    """

    def __init__(
            self,
            input_shape: int,
            T: float = -0.2,
            learnable: bool = True,
            inplace: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(
                -0.2 * torch.ones(input_shape)) if learnable else tensor

        self.T: Tensor = create_param(T)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        result = torch.nn.functional.relu(x) * torch.nn.functional.sigmoid(x) + self.T
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result


@register_activation
class PFPM(BaseActivation):
    """
    Parametric Flatten-p Mish (PFPM).

    PFPM is an Adaptive Activation Function (AAF).

    The PFPM activation function is defined as:

    .. math:: PFPM(z_i) =
    \begin{cases}
    z_i \tanh(\ln(1 + \exp(z_i))) + p_i, & z_i \geq 0, \\
    p_i, & z_i < 0,
    \end{cases}

    where:
    - z_i is the input to the activation function for neuron i.
    - p_i is a trainable parameter for neuron i.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        p_i (float, optional): Initial value for parameter `p_i`. Default is 0.0.
        learnable (bool, optional): Whether `p_i` is trainable. Default is True.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for BaseActivation.

    Attributes:
        p_i (Tensor or nn.Parameter): Trainable or fixed parameter `p_i`.
        inplace (bool): Whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the Parametric Flatten-p Mish transformation.
    """

    def __init__(
            self,
            input_shape: int,
            p: float = 1.0,
            learnable: bool = True,
            inplace: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        def create_param(value: float) -> Tensor:
            tensor = torch.full((input_shape, 1), value, dtype=torch.float64)
            return nn.Parameter(torch.randn(input_shape)) if learnable else tensor

        self.p: Tensor = create_param(p)
        self.inplace: bool = inplace

    def _forward(self, x: Tensor) -> Tensor:
        func = x * torch.tanh(torch.log(1 + torch.exp(x))) + self.p
        result = torch.nn.functional.relu(x) * func + self.p
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result


@register_activation
class PSIGRAMP(BaseActivation):
    """
    Parametric Sigmoid-Ramp (P-SIG-RAMP).

    P-SIG-RAMP is an Adaptive Activation Function (AAF) that combines the logistic sigmoid and a piecewise linear function.

    The P-SIG-RAMP activation function is defined as:

    .. math::
        f(z_i) = a_i \sigma(z_i) + (1 - a_i) \cdot
        \begin{cases}
        1, & z_i \geq \frac{1}{2b_i}, \\
        b_i z_i + \frac{1}{2}, & -\frac{1}{2b_i} < z_i < \frac{1}{2b_i}, \\
        0, & z_i \leq -\frac{1}{2b_i},
        \end{cases}

    where:
    - \( z_i \) is the input to the activation function for neuron \( i \).
    - \( a_i \) is a trainable parameter constrained to \( [0,1] \).
    - \( b_i \) is a trainable parameter.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a_i (float, optional): Initial value for parameter \( a_i \). Default is 0.5.
        b_i (float, optional): Initial value for parameter \( b_i \). Default is 1.0.
        learnable (bool, optional): Whether \( a_i \) and \( b_i \) are trainable. Default is True.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for BaseActivation.

    Attributes:
        a_i (Tensor or nn.Parameter): Trainable or fixed parameter \( a_i \).
        b_i (Tensor or nn.Parameter): Trainable or fixed parameter \( b_i \).
        inplace (bool): Whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the Parametric Sigmoid-Ramp transformation.
    """

    def __init__(
            self,
            input_shape: int,
            a: float = 0.5,
            b: float = 1.0,
            learnable: bool = True,
            inplace: bool = False,
            **kwargs
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
                    torch.nn.functional.relu(2 * self.b * x + 1) - torch.nn.functional.relu(2 * self.b * x - 1))

        result = converted_a * (torch.nn.functional.sigmoid(x)) + (1 - converted_a) * relu_section
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result




@register_activation
class RSIGN(BaseActivation):
    """
    React-Sign (RSign).

    RSign is an Adaptive Activation Function (AAF) that introduces an adaptive threshold to the standard sign function.

    The RSign activation function is defined as:

    .. math::
        f(z_i) =
        \begin{cases}
        1, & z_i \geq a_c, \\
        -1, & z_i < a_c,
        \end{cases}

    where:
    - \( z_i \) is the input to the activation function for neuron \( i \).
    - \( a_c \) is an adaptive threshold parameter for each channel.

    Args:
        input_shape (int): Size of the input vector tensor (feature size).
        a_c (float, optional): Initial value for threshold \( a_c \). Default is 0.0.
        learnable (bool, optional): Whether \( a_c \) is trainable. Default is True.
        inplace (bool, optional): Whether to perform operations in-place. Default is False.
        **kwargs: Additional keyword arguments for BaseActivation.

    Attributes:
        a_c (Tensor or nn.Parameter): Trainable or fixed parameter \( a_c \).
        inplace (bool): Whether operations are performed in-place.

    Methods:
        _forward(x: Tensor) -> Tensor:
            Computes the React-Sign transformation.
    """

    def __init__(
            self,
            input_shape: int,
            a: float = 0.5,
            learnable: bool = True,
            inplace: bool = False,
            **kwargs
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
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        return result


