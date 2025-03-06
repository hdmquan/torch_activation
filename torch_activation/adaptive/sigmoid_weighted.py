import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from torch_activation import register_activation


@register_activation
class Swish(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        result = x * torch.sigmoid(self.a * x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class AHAF(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        result = self.a * x * torch.sigmoid(self.b * x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSSiLU(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Ensure b is less than 1 to avoid division by zero
            self.b = nn.Parameter(Tensor([min(b, 0.99)]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([min(b, 0.99)])  # Ensure b is less than 1

    def forward(self, x) -> Tensor:
        # Compute the shifted and normalized sigmoid
        shifted_sigmoid = (torch.sigmoid(self.a * x) - self.b) / (1 - self.b)
        result = x * shifted_sigmoid
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ESwish(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        result = self.a * x * torch.sigmoid(x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ACONB(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            # Ensure b is between 0 and 1
            self.b = nn.Parameter(Tensor([max(0.0, min(b, 1.0))]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([max(0.0, min(b, 1.0))])  # Ensure b is between 0 and 1

    def forward(self, x) -> Tensor:
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
class ACONC(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
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
class PSGU(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        result = x * torch.tanh(self.a * torch.sigmoid(x))
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TBSReLUl(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # Calculate bipolar sigmoid: (1 - exp(-x)) / (1 + exp(-x))
        bipolar_sigmoid = (1 - torch.exp(-x)) / (1 + torch.exp(-x))
        result = x * torch.tanh(self.a * bipolar_sigmoid)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PATS(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        if learnable:
            # Initialize with a value in the valid range
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
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
class AQuLU(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            # Ensure a is positive to avoid division by zero
            self.a = nn.Parameter(Tensor([max(1e-6, a)]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([max(1e-6, a)])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
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
            result[mask_middle] = self.a * x[mask_middle]**2 + self.b * x[mask_middle]
            result[mask_lower] = 0
            
            # Copy back to original tensor
            x.copy_(result)
            return x
        else:
            # Initialize result tensor
            result = torch.zeros_like(x)
            
            # Apply different functions to different regions
            result[mask_upper] = x[mask_upper]
            result[mask_middle] = self.a * x[mask_middle]**2 + self.b * x[mask_middle]
            # result[mask_lower] is already 0
            
            return result


@register_activation
class SinLU(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        modified_x = x + self.a * torch.sin(self.b * x)
        result = modified_x * torch.sigmoid(x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ErfAct(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Calculate exp(b*x) with clipping to prevent overflow
        exp_term = torch.exp(torch.clamp(self.b * x, max=20))
        result = x * torch.erf(self.a * exp_term)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSerf(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Calculate softplus: ln(1 + exp(b*x))
        softplus = torch.log(1 + torch.exp(torch.clamp(self.b * x, max=20)))
        result = x * torch.erf(self.a * softplus)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class Swim(nn.Module):
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
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
        else:
            self.a = Tensor([a])

    def forward(self, x) -> Tensor:
        # Calculate the modified sigmoid-like term
        sigmoid_term = 0.5 * (1 + (self.a * x) / torch.sqrt(1 + x.pow(2)))
        result = x * sigmoid_term
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result