import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch_activation import register_activation

@register_activation
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

@register_activation
class LAAF(nn.Module):
    r"""
    Applies the Locally Adaptive Activation Function (LAAF):

    :math:`\text{LAAF}(x) = g(a \cdot x)`

    where :math:`a` is a trainable parameter for each neuron and :math:`g` is any activation function.

    See: https://doi.org/10.1016/j.cma.2020.113028

    Args:
        activation (str, optional): The activation function to use. Options: 'sigmoid', 'tanh', 'relu', 'leaky_relu'. Default: 'sigmoid'
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0
        leaky_slope (float, optional): Leakiness parameter for LeakyReLU. Default: 0.01
        fixed_n (float, optional): Fixed parameter to accelerate convergence. If > 1, applies g(n*a*x). Default: 1.0
        inplace (bool, optional): Can optionally do the operation in-place when possible. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LAAF(activation='tanh', a_init=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = LAAF(activation='relu', fixed_n=2.0, inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, activation: str = 'sigmoid', a_init: float = 1.0, 
                 leaky_slope: float = 0.01, fixed_n: float = 1.0, inplace: bool = False):
        super(LAAF, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.activation = activation.lower()
        self.leaky_slope = leaky_slope
        self.fixed_n = fixed_n
        self.inplace = inplace
        
        if self.activation not in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
            raise ValueError(f"Unsupported activation: {activation}. Choose from 'sigmoid', 'tanh', 'relu', 'leaky_relu'")

    def forward(self, x) -> Tensor:
        scaled_x = self.fixed_n * self.a * x
        
        if self.activation == 'sigmoid':
            return torch.sigmoid(scaled_x)
        elif self.activation == 'tanh':
            return torch.tanh(scaled_x)
        elif self.activation == 'relu':
            return F.relu(scaled_x, inplace=self.inplace)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(scaled_x, negative_slope=self.leaky_slope, inplace=self.inplace)


@register_activation
class AdaptiveSlopeTanh(nn.Module):
    r"""
    Applies the Adaptive Slope Hyperbolic Tangent function:

    :math:`\text{AdaptiveSlopeTanh}(x) = \tanh(a \cdot x)`

    where :math:`a` is a trainable parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = AdaptiveSlopeTanh(a_init=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 1.0):
        super(AdaptiveSlopeTanh, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))

    def forward(self, x) -> Tensor:
        return torch.tanh(self.a * x)


@register_activation
class PSTanh(nn.Module):
    r"""
    Applies the Parametric Scaled Hyperbolic Tangent function:

    :math:`\text{PSTanh}(x) = x \cdot a \cdot (1 + \tanh(b \cdot x))`

    where :math:`a` and :math:`b` are trainable parameters.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0
        b_init (float, optional): Initial value for the trainable parameter b. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = PSTanh(a_init=2.0, b_init=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 1.0):
        super(PSTanh, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.b = nn.Parameter(Tensor([b_init]))

    def forward(self, x) -> Tensor:
        return x * self.a * (1 + torch.tanh(self.b * x))


@register_activation
class SSinH(nn.Module):
    r"""
    Applies the Scaled Sine-Hyperbolic function:

    :math:`\text{SSinH}(x) = a \cdot \sinh(b \cdot x)`

    where :math:`a` and :math:`b` are trainable parameters.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0
        b_init (float, optional): Initial value for the trainable parameter b. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SSinH(a_init=2.0, b_init=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 1.0):
        super(SSinH, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.b = nn.Parameter(Tensor([b_init]))

    def forward(self, x) -> Tensor:
        return self.a * torch.sinh(self.b * x)


@register_activation
class SExp(nn.Module):
    r"""
    Applies the Scaled Exponential function:

    :math:`\text{SExp}(x) = a \cdot (\exp(b \cdot x) - 1)`

    where :math:`a` and :math:`b` are trainable parameters.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0
        b_init (float, optional): Initial value for the trainable parameter b. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SExp(a_init=2.0, b_init=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 1.0):
        super(SExp, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.b = nn.Parameter(Tensor([b_init]))

    def forward(self, x) -> Tensor:
        return self.a * (torch.exp(self.b * x) - 1)


@register_activation
class LAU(nn.Module):
    r"""
    Applies the Logmoid Activation Unit function:

    :math:`\text{LAU}(x) = x \cdot \ln(1 + a \cdot \sigma(b \cdot x))`

    where :math:`a` and :math:`b` are trainable parameters and :math:`\sigma` is the sigmoid function.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0
        b_init (float, optional): Initial value for the trainable parameter b. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LAU(a_init=2.0, b_init=1.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 1.0):
        super(LAU, self).__init__()
        self.a = nn.Parameter(Tensor([a_init]))
        self.b = nn.Parameter(Tensor([b_init]))

    def forward(self, x) -> Tensor:
        return x * torch.log(1 + self.a * torch.sigmoid(self.b * x))


@register_activation
class AGumb(nn.Module):
    r"""
    Applies the Adaptive Gumbel function:

    :math:`\text{AGumb}(x) = 1 - (1 + a \cdot \exp(x))^{-1}`

    where :math:`a` is a trainable positive parameter.

    Args:
        a_init (float, optional): Initial value for the trainable parameter a. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = AGumb(a_init=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a_init: float = 1.0):
        super(AGumb, self).__init__()
        # Using softplus to ensure a is positive
        self.a_raw = nn.Parameter(Tensor([a_init]))

    def forward(self, x) -> Tensor:
        # Ensure a is positive using softplus
        a = F.softplus(self.a_raw)
        return 1 - (1 + a * torch.exp(x))**(-1)
