import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable

from torch_activation import register_activation


@register_activation
class AdaptiveSigmoid(nn.Module):
    r"""
    Applies the Adaptive Sigmoid function:

    :math:`\text{AdaptiveSigmoid}(x) = \frac{2}{1 - \exp(-ax)} - \frac{2}{a(1 + \exp(-ax))}`

    where :math:`a \in (0, \infty)` is a learnable parameter.

    Args:
        a (float, optional): Slope parameter. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.AdaptiveSigmoid(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.AdaptiveSigmoid(learnable=True)
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
            # Ensure a is positive
            self.a = nn.Parameter(Tensor([abs(a)]))
        else:
            self.a = Tensor([abs(a)])

    def forward(self, x) -> Tensor:
        # Compute the adaptive sigmoid
        term1 = 2 / (1 - torch.exp(-self.a * x))
        term2 = 2 / (self.a * (1 + torch.exp(-self.a * x)))
        result = term1 - term2
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class GeneralizedHyperbolicTangent(nn.Module):
    r"""
    Applies the Generalized Hyperbolic Tangent function:

    :math:`\text{GeneralizedHyperbolicTangent}(x) = a \cdot \frac{1 - \exp(-b \cdot x)}{1 + \exp(-b \cdot x)}`

    Args:
        a (float, optional): Amplitude parameter. Default: 1.0
        b (float, optional): Slope parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.GeneralizedHyperbolicTangent(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.GeneralizedHyperbolicTangent(learnable=True)
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
        # Compute the generalized hyperbolic tangent
        numerator = 1 - torch.exp(-self.b * x)
        denominator = 1 + torch.exp(-self.b * x)
        result = self.a * (numerator / denominator)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TrainableAmplitude(nn.Module):
    r"""
    Applies the Trainable Amplitude function:

    :math:`\text{TrainableAmplitude}(x) = a \cdot g(x) + b`

    where :math:`g(x)` is a base activation function.

    Args:
        base_activation (callable, optional): Base activation function. Default: ``torch.tanh``
        a (float, optional): Amplitude parameter. Default: 1.0
        b (float, optional): Bias parameter. Default: 0.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TrainableAmplitude(base_activation=torch.tanh, a=1.5, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TrainableAmplitude(base_activation=torch.sigmoid, learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        base_activation: Callable = torch.tanh,
        a: float = 1.0, 
        b: float = 0.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.base_activation = base_activation
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([a]))
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([a])
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Apply the base activation function
        activated = self.base_activation(x)
        
        # Scale and shift
        result = self.a * activated + self.b
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class ASSF(nn.Module):
    r"""
    Applies the Adaptive Slope Sigmoidal Function:

    :math:`\text{ASSF}(x) = \sigma(a \cdot x) = \frac{1}{1 + \exp(-a \cdot x)}`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Slope parameter. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ASSF(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ASSF(learnable=True)
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
        # Compute the adaptive slope sigmoid
        result = torch.sigmoid(self.a * x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class SVAF(nn.Module):
    r"""
    Applies the Slope Varying Activation Function:

    :math:`\text{SVAF}(x) = \tanh(a \cdot x)`

    Args:
        a (float, optional): Slope parameter. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SVAF(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SVAF(learnable=True)
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
        # Compute the slope varying tanh
        result = torch.tanh(self.a * x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TanhSoft(nn.Module):
    r"""
    Applies the TanhSoft function:

    :math:`\text{TanhSoft}(x) = \tanh(a \cdot x + b \cdot \exp(c \cdot x)) \cdot \ln(d + \exp(x))`

    where :math:`a \in (-\infty, 1]`, :math:`b \in [0, \infty)`, :math:`c \in (0, \infty)`, :math:`d \in [0, 1]`.

    Args:
        a (float, optional): Parameter for linear term. Default: 0.5
        b (float, optional): Parameter for exponential term. Default: 0.5
        c (float, optional): Parameter for exponential scaling. Default: 1.0
        d (float, optional): Parameter for logarithmic term. Default: 0.1
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TanhSoft(a=0.5, b=0.5, c=1.0, d=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TanhSoft(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        a: float = 0.5, 
        b: float = 0.5, 
        c: float = 1.0,
        d: float = 0.1,
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            # Constrain parameters to their valid ranges
            self.a = nn.Parameter(Tensor([min(a, 1.0)]))
            self.b = nn.Parameter(Tensor([max(b, 0.0)]))
            self.c = nn.Parameter(Tensor([max(c, 0.01)]))  # Avoid c=0
            self.d = nn.Parameter(Tensor([max(min(d, 1.0), 0.0)]))
        else:
            self.a = Tensor([min(a, 1.0)])
            self.b = Tensor([max(b, 0.0)])
            self.c = Tensor([max(c, 0.01)])  # Avoid c=0
            self.d = Tensor([max(min(d, 1.0), 0.0)])

    def forward(self, x) -> Tensor:
        # Compute the TanhSoft function
        tanh_term = torch.tanh(self.a * x + self.b * torch.exp(self.c * x))
        log_term = torch.log(self.d + torch.exp(x))
        result = tanh_term * log_term
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TanhSoft1(nn.Module):
    r"""
    Applies the TanhSoft-1 function:

    :math:`\text{TanhSoft-1}(x) = \tanh(a \cdot x) \cdot \ln(1 + \exp(x))`

    Args:
        a (float, optional): Slope parameter for tanh. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TanhSoft1(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TanhSoft1(learnable=True)
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
        # Compute the TanhSoft-1 function
        tanh_term = torch.tanh(self.a * x)
        softplus_term = torch.log(1 + torch.exp(x))
        result = tanh_term * softplus_term
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TanhSoft2(nn.Module):
    r"""
    Applies the TanhSoft-2 function:

    :math:`\text{TanhSoft-2}(x) = x \cdot \tanh(b \cdot \exp(c \cdot x))`

    Args:
        b (float, optional): Amplitude parameter. Default: 1.0
        c (float, optional): Exponential scaling parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TanhSoft2(b=1.0, c=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TanhSoft2(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        b: float = 1.0, 
        c: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.b = nn.Parameter(Tensor([b]))
            self.c = nn.Parameter(Tensor([c]))
        else:
            self.b = Tensor([b])
            self.c = Tensor([c])

    def forward(self, x) -> Tensor:
        # Compute the TanhSoft-2 function
        tanh_term = torch.tanh(self.b * torch.exp(self.c * x))
        result = x * tanh_term
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class TanhSoft3(nn.Module):
    r"""
    Applies the TanhSoft-3 function:

    :math:`\text{TanhSoft-3}(x) = \ln(1 + \exp(x) \cdot \tanh(a \cdot x))`

    Args:
        a (float, optional): Slope parameter for tanh. Default: 1.0
        learnable (bool, optional): optionally make ``a`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.TanhSoft3(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TanhSoft3(learnable=True)
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
        # Compute the TanhSoft-3 function
        inner_term = torch.exp(x) * torch.tanh(self.a * x)
        result = torch.log(1 + inner_term)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSigmoid(nn.Module):
    r"""
    Applies the Parametric Sigmoid function:

    :math:`\text{PSigmoid}(x) = a \cdot \sigma(b \cdot x)`

    where :math:`\sigma(x)` is the sigmoid function.

    Args:
        a (float, optional): Amplitude parameter. Default: 1.0
        b (float, optional): Slope parameter. Default: 1.0
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PSigmoid(a=1.5, b=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSigmoid(learnable=True)
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
        # Compute the parametric sigmoid
        result = self.a * torch.sigmoid(self.b * x)
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class PSF(nn.Module):
    r"""
    Applies the Parametric Sigmoid Function:

    :math:`\text{PSF}(x) = \frac{1}{(1 + \exp(-x))^m}`

    Args:
        m (float, optional): Power parameter. Default: 1.0
        learnable (bool, optional): optionally make ``m`` trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.PSF(m=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PSF(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        m: float = 1.0, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.m = nn.Parameter(Tensor([m]))
        else:
            self.m = Tensor([m])

    def forward(self, x) -> Tensor:
        # Compute the parametric sigmoid function
        sigmoid = 1 / (1 + torch.exp(-x))
        result = sigmoid ** self.m
        
        if self.inplace and hasattr(x, 'copy_'):
            x.copy_(result)
            return x
        else:
            return result


@register_activation
class STACTanh(nn.Module):
    r"""
    Applies the Slope and Threshold Adaptive Activation Function with tanh:

    :math:`\text{STAC-tanh}(x) = \begin{cases} 
        \tanh(-a) + b \cdot (x + a), & x < -a \\
        \tanh(x), & -a \leq x \leq a \\
        \tanh(a) + b \cdot (x - a), & x > a 
    \end{cases}`

    Args:
        a (float, optional): Threshold parameter. Default: 1.0
        b (float, optional): Slope parameter for linear regions. Default: 0.1
        learnable (bool, optional): optionally make parameters trainable. Default: ``False``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.STACTanh(a=1.0, b=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.STACTanh(learnable=True)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 0.1, 
        learnable: bool = False, 
        inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.a = nn.Parameter(Tensor([abs(a)]))  # Ensure a is positive
            self.b = nn.Parameter(Tensor([b]))
        else:
            self.a = Tensor([abs(a)])  # Ensure a is positive
            self.b = Tensor([b])

    def forward(self, x) -> Tensor:
        # Precompute constants
        tanh_a = torch.tanh(self.a)
        tanh_neg_a = torch.tanh(-self.a)
        
        # Create masks for different regions
        mask_lower = x < -self.a
        mask_middle = (x >= -self.a) & (x <= self.a)
        mask_upper = x > self.a
        
        if self.inplace:
            # Create a copy to avoid modifying during computation
            result = x.clone()
            
            # Apply different functions to different regions
            result[mask_lower] = tanh_neg_a + self.b * (x[mask_lower] + self.a)
            result[mask_middle] = torch.tanh(x[mask_middle])
            result[mask_upper] = tanh_a + self.b * (x[mask_upper] - self.a)
            
            # Copy back to original tensor
            x.copy_(result)
            return x
        else:
            # Initialize result tensor
            result = torch.zeros_like(x)
            
            # Apply different functions to different regions
            result[mask_lower] = tanh_neg_a + self.b * (x[mask_lower] + self.a)
            result[mask_middle] = torch.tanh(x[mask_middle])
            result[mask_upper] = tanh_a + self.b * (x[mask_upper] - self.a)
            
            return result