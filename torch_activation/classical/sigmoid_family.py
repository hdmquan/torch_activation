import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch_activation import register_activation
from torch_activation.utils import sech
# TODO: Optimize any functions that use where

@register_activation
class Sigmoid(nn.Module):
    r"""
    Applies the Sigmoid activation function:

    :math:`\text{Sigmoid}(z) = \frac{1}{1 + \exp(-z)}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Sigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.Sigmoid(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()

        self.inplace = inplace

    def forward(self, z) -> Tensor:
        if self.inplace:
            z.sigmoid_()
            return z
        else:
            return torch.sigmoid(z)


@register_activation
class Tanh(nn.Module):
    r"""
    Applies the Tanh activation function:

    :math:`\text{Tanh}(z) = \tanh(z)`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Tanh()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.Tanh(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        if self.inplace:
            z.tanh_()
            return z
        else:
            return torch.tanh(z)


@register_activation
class ShiftedScaledSigmoid(nn.Module):
    r"""
    Applies the Shifted Scaled Sigmoid activation function:

    :math:`\text{ShiftedScaledSigmoid}(z) = \frac{1}{1 + \exp(-a(z-b))}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Shift parameter. Default: 0.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.ShiftedScaledSigmoid(a=2.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.ShiftedScaledSigmoid(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, inplace: bool = False):
        super(ShiftedScaledSigmoid, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return torch.sigmoid(self.a * (z - self.b))

    def _forward_inplace(self, z):
        z.sub_(self.b).mul_(self.a)
        z.sigmoid_()
        return z


@register_activation
class VariantSigmoidFunction(nn.Module):
    r"""
    Applies the Variant Sigmoid Function activation:

    :math:`\text{VariantSigmoidFunction}(z) = \frac{a}{1 + \exp(-bz)} - c`

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Slope parameter. Default: 1.0
        c (float, optional): Offset parameter. Default: 0.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.VariantSigmoidFunction(a=2.0, b=1.5, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.VariantSigmoidFunction(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(
        self, a: float = 1.0, b: float = 1.0, c: float = 0.0, inplace: bool = False
    ):
        super(VariantSigmoidFunction, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.c = nn.Parameter(torch.tensor([c]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return self.a * torch.sigmoid(self.b * z) - self.c

    def _forward_inplace(self, z):
        z.mul_(self.b).sigmoid_().mul_(self.a).sub_(self.c)
        return z


@register_activation
class STanh(nn.Module):
    r"""
    Applies the Scaled Hyperbolic Tangent activation function:

    :math:`\text{STanh}(z) = a \tanh(bz)`

    :note: Lecun et al. (1998) suggested that the scaling factor \( a \) should be 1.7159 and the slope parameter \( b \) should be 2/3.

    Args:
        a (float, optional): Scale parameter. Default: 1.7159
        b (float, optional): Slope parameter. Default: 2/3
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.STanh(a=1.7, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.STanh(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.7159, b: float = 2 / 3, inplace: bool = False):
        super(STanh, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return self.a * torch.tanh(self.b * z)

    def _forward_inplace(self, z):
        z.mul_(self.b).tanh_().mul_(self.a)
        return z


# FIXME: Revise this
# @register_activation
# class BiModalDerivativeSigmoid(nn.Module):
#     r"""
#     Applies the Bi-Modal Derivative Sigmoid activation function:

#     :math:`\text{BiModalDerivativeSigmoid}(z) = \frac{a}{1 + \exp(-bz)} - \frac{1}{2} \left( \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z-b)} \right)`

#     Args:
#         b (float, optional): Shift parameter. Default: 1.0

#     Shape:
#         - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
#         - Output: :math:`(*)`, same shape as the input.

#     Examples::

#         >>> m = tac.BiModalDerivativeSigmoid(b=2.0)
#         >>> x = torch.randn(2)
#         >>> output = m(x)
#     """

#     def __init__(self, b: float = 1.0):
#         super(BiModalDerivativeSigmoid, self).__init__()
#         self.b = nn.Parameter(torch.tensor([b]))

#     def forward(self, z):
#         first_term = self.a * torch.sigmoid(self.b * z)
#         second_term = torch.sigmoid(z) + torch.sigmoid(z + self.b)
#         return 0.5 * (first_term - second_term)


@register_activation
class Arctan(nn.Module):
    r"""
    Applies the Arctan activation function:

    :math:`\text{Arctan}(z) = \arctan(z)`

    The arctangent function resembles a logistic sigmoid activation but covers a wider range
    :math:`[-\frac{\pi}{2}, \frac{\pi}{2}]`. It was initially used as an activation function
    over twenty years ago and was rediscovered in more recent research where it showed
    competitive performance compared to tanh, ReLU, leaky ReLU, logistic sigmoid, and swish
    activation functions.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Arctan()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(Arctan, self).__init__()

    def forward(self, z) -> Tensor:
        return torch.atan(z)


@register_activation
class ArctanGR(nn.Module):
    r"""
    Applies the ArctanGR activation function:

    :math:`\text{ArctanGR}(z) = \frac{\arctan(z)}{1 + \sqrt{2}}`

    ArctanGR is a scaled version of the Arctan activation function. The scaling factor
    :math:`\frac{1}{1 + \sqrt{2}}` was found to be particularly effective in experiments,
    outperforming other activation functions including the standard Arctan. Other scaling
    variants such as division by :math:`\pi`, :math:`\frac{1 + \sqrt{5}}{2}` (golden ratio),
    or the Euler number have also been explored in the literature.

    Args:
        scale_factor (float): The scaling factor for the arctangent output.
            Default: :math:`\frac{1}{1 + \sqrt{2}} \approx 0.2929`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.ArctanGR()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> # With custom scale factor
        >>> m = tac.ArctanGR(scale_factor=1/math.pi)
        >>> output = m(x)
    """

    def __init__(
        self, scale_factor: float = 1.0 / (1.0 + torch.sqrt(torch.tensor(2.0)))
    ):
        super(ArctanGR, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, z) -> Tensor:
        return torch.atan(z) * self.scale_factor


@register_activation
class SigmoidAlgebraic(nn.Module):
    r"""
    Applies the Sigmoid Algebraic activation function:

    :math:`\text{SigmoidAlgebraic}(z) = \frac{1}{1 + \exp\left(-\frac{z(1 + a|z|)}{1 + |z|(1 + a|z|)}\right)}`
    :note: \( a > 0 \).
    Args:
        a (float, optional): Shape parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SigmoidAlgebraic(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.SigmoidAlgebraic(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0):
        super(SigmoidAlgebraic, self).__init__()
        assert a > 0, "a must be greater than 0"
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        abs_z = torch.abs(z)
        a_abs_z = self.a * abs_z
        numerator = z * (1 + a_abs_z)
        denominator = 1 + abs_z * (1 + a_abs_z)
        # Sigmoid is 1 on (1 plus e to the power of -z) :(
        return torch.sigmoid(numerator / denominator)


@register_activation
class TripleStateSigmoid(nn.Module):
    r"""
    Applies the Triple State Sigmoid activation function:

    :math:`\text{TripleStateSigmoid}(z) = \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z+a)} + \frac{1}{1 + \exp(-z+b)}`

    :note: The default values of \( a \) and \( b \) are 20.0 and 40.0, respectively, as suggested in the paper.
    (https://www.sciencedirect.com/science/article/abs/pii/S0957417420307557).

    Args:
        a (float, optional): First shift parameter. Default: 20.0
        b (float, optional): Second shift parameter. Default: 40.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.TripleStateSigmoid()  # Uses default values a=20.0, b=40.0 from the paper
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = tac.TripleStateSigmoid(a=15.0, b=30.0)  # Custom parameters
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 20.0, b: float = 40.0, inplace: bool = False):
        super(TripleStateSigmoid, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return torch.sigmoid(z) + torch.sigmoid(z - self.a) + torch.sigmoid(z - self.b)


@register_activation
class ImprovedLogisticSigmoid(nn.Module):
    r"""
    Applies the Improved Logistic Sigmoid activation function:

    :math:`\text{ImprovedLogisticSigmoid}(z) = \begin{cases} 
    a(z-b) + \sigma(b), & z \geq b \\ 
    \sigma(z), & -b < z < b \\ 
    a(z+b) + \sigma(-b), & z \leq -b 
    \end{cases}`

    This activation function was designed to address the vanishing gradient problem 
    of the standard logistic sigmoid. It behaves like the standard sigmoid in the middle region 
    but has a linear response in the saturation regions, allowing for non-zero gradients 
    even for large input magnitudes.

    The parameter 'a' controls the slope of the linear regions and should satisfy:
    :math:`a > a_{min} = \frac{\exp(-b)}{(1 + \exp(-b))^2}`

    This ensures the function remains smooth at the transition points.

    The output range is :math:`(-\infty, \infty)`, unlike the standard sigmoid which is bounded 
    to :math:`(0, 1)`. Research has shown this activation function has higher convergence speed 
    than the standard logistic sigmoid.

    Args:
        a (float, optional): Slope parameter for the linear regions. Default: 0.2
        b (float, optional): Threshold parameter defining the transition points. Default: 2.0
        trainable (bool, optional): Whether parameters a and b should be trainable. Default: False

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.ImprovedLogisticSigmoid(a=0.1, b=3.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> # With trainable parameters
        >>> m = tac.ImprovedLogisticSigmoid(trainable=True)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.2, b: float = 2.0, trainable: bool = False):
        super(ImprovedLogisticSigmoid, self).__init__()

        b_tensor = torch.tensor([b])
        a_min = torch.exp(-b_tensor) / (1 + torch.exp(-b_tensor)) ** 2

        a = max(a, a_min.item() * 1.01)  # Add 1% margin

        if trainable:
            self.a = nn.Parameter(torch.tensor([a]))
            self.b = nn.Parameter(torch.tensor([b]))
        else:
            self.register_buffer("a", torch.tensor([a]))
            self.register_buffer("b", torch.tensor([b]))

    def forward(self, z) -> Tensor:
        sig_b = torch.sigmoid(self.b)
        
        upper_mask = z >= self.b
        lower_mask = z <= -self.b
        
        result = torch.sigmoid(z)

        if self.a != 0:  # To not compute linear extensions where not needed
            upper_region = self.a * (z - self.b) + sig_b
            lower_region = self.a * (z + self.b) + (1 - sig_b)
            
            # Apply masks
            result = torch.where(upper_mask, upper_region, result)
            result = torch.where(lower_mask, lower_region, result)
        
        return result


@register_activation
class SigLin(nn.Module):
    r"""
    Applies the SigLin activation function:

    :math:`\text{SigLin}(z) = \sigma(z) + az`

    :note: The authors of the study (https://link.springer.com/article/10.1007/s13748-020-00218-y) evaluated the SigLin activation function using linear coefficients of 0, 0.05, 0.1, and 0.15.

    Args:
        a (float, optional): Linear coefficient. Default: 0.1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SigLin(a=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

    """

    def __init__(self, a: float = 0.1):
        super(SigLin, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        return torch.sigmoid(z) + self.a * z


@register_activation
class PTanh(nn.Module):
    r"""
    Applies the Penalized Hyperbolic Tangent activation function:

    :math:`\text{PTanh}(z) = \begin{cases} 
    \tanh(z), & z \geq 0 \\ 
    \frac{\tanh(z)}{a}, & z < 0 
    \end{cases}`

    :note: a must be greater than 1.0

    Args:
        a (float, optional): Penalty factor for negative inputs. Default: 2.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.PTanh(a=3.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 2.0):
        super(PTanh, self).__init__()
        assert a > 1.0, "a must be greater than 1.0"
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

    def forward(self, z) -> Tensor:
        # Turned the thing from 1 line to 10 just to not use where :D
        tanh_z = torch.tanh(z)
        
        neg_mask = z < 0
        
        result = tanh_z.clone()

        if neg_mask.any():
            if self.a != 0:
                result[neg_mask] = tanh_z[neg_mask] / self.a
            else:
                result[neg_mask] = float('inf') * torch.sign(tanh_z[neg_mask])
        
        return result



@register_activation
class SRS(nn.Module):
    r"""
    Applies the Soft Root Sign activation function:

    :math:`\text{SRS}(z) = \frac{z}{\sqrt[a]{1 + \exp\left(-\frac{z}{b}\right)}}`

    Args:
        a (float, optional): Root parameter. Default: 2.0
        b (float, optional): Scale parameter. Default: 3.0
        learnable (bool, optional): If True, the parameters are learnable. Default: False

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SRS(a=3.0, b=0.5, learnable=True)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 2.0, b: float = 3.0, learnable: bool = False):
        super(SRS, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.learnable = learnable

        if not learnable:
            self.a.requires_grad = False
            self.b.requires_grad = False

    def forward(self, z) -> Tensor:
        denominator = torch.pow(1 + torch.exp(-z / self.b), 1 / self.a)
        return z / denominator



@register_activation
class SC(nn.Module):
    r"""
    Applies the Soft Clipping activation function:

    :math:`\text{SC}(z) = \frac{1}{a} \ln\left(\frac{1 + \exp(az)}{1 + \exp(a(z-1))}\right)`
    :note: ReLU1 but soft edges

    Args:
        a (float, optional): Sharpness parameter. Default: 50.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SC(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 50.0):
        super(SC, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

    def forward(self, z) -> Tensor:
        numerator = 1 + torch.exp(self.a * z)
        denominator = 1 + torch.exp(self.a * (z - 1))
        return torch.log(numerator / denominator) / self.a



@register_activation
class Hexpo(nn.Module):
    r"""
    Applies the Hexpo activation function:

    :math:`\text{Hexpo}(z) = \begin{cases} 
    -a \exp\left(-\frac{z}{b}\right) - 1, & z \geq 0 \\ 
    c \exp\left(-\frac{z}{d}\right) - 1, & z < 0 
    \end{cases}`

    :note: a, b, c and d could be trainable parameters, but could lead to vanishing gradients

    Args:
        a (float, optional): Positive scale parameter. Default: 1.0
        b (float, optional): Positive decay parameter. Default: 1.0
        c (float, optional): Negative scale parameter. Default: 1.0
        d (float, optional): Negative decay parameter. Default: 1.0
        learnable (bool, optional): If True, the parameters are learnable. Default: False (recommended)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Hexpo(a=1.5, b=0.5, c=2.0, d=0.7)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        learnable: bool = False,
    ):
        super(Hexpo, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.c = nn.Parameter(torch.tensor([c]))
        self.d = nn.Parameter(torch.tensor([d]))
        self.learnable = learnable

        if not learnable:
            self.a.requires_grad = False
            self.b.requires_grad = False
            self.c.requires_grad = False
            self.d.requires_grad = False

    def forward(self, z) -> Tensor:
        pos_mask = z >= 0
        
        result = torch.empty_like(z)
        
        if pos_mask.any():
            result[pos_mask] = -self.a * torch.exp(-z[pos_mask] / self.b) - 1
        
        neg_mask = ~pos_mask
        if neg_mask.any():
            result[neg_mask] = self.c * torch.exp(-z[neg_mask] / self.d) - 1
        
        return result


@register_activation
class Softsign(nn.Module):
    r"""
    Applies the Softsign activation function:

    :math:`\text{Softsign}(z) = \frac{z}{1 + |z|}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Softsign()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(Softsign, self).__init__()

    def forward(self, z) -> Tensor:
        return z / (1 + torch.abs(z))


@register_activation
class SmoothStep(nn.Module):
    r"""
    Applies the Smooth Step activation function:

    :math:`\text{SmoothStep}(z) = \begin{cases} 
    1, & z \geq \frac{a}{2} \\ 
    \frac{2}{a^3} z^3 - \frac{3}{2a} z + \frac{1}{2}, & -\frac{a}{2} \leq z \leq \frac{a}{2} \\ 
    0, & z \leq -\frac{a}{2} 
    \end{cases}`

    Args:
        a (float, optional): Width parameter. Default: 2.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SmoothStep(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(SmoothStep, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

    def forward(self, z) -> Tensor:
        half_a = self.a / 2
        
        result = torch.ones_like(z)
        
        middle_mask = (z > -half_a) & (z < half_a)
        lower_mask = z <= -half_a

        if middle_mask.any(): 
            # Compute polynomial using Horner
            # Original: cubic_term - linear_term + constant_term
            # = (2 / (a^3)) * z^3 - (3 / (2 * a)) * z + 0.5
            # Horner: 0.5 + z*(-3/(2*a) + z*z*(2/(a^3)))
            z_middle = z[middle_mask]
            inv_a = 1.0 / self.a
            
            middle_result = 0.5 + z_middle * (-1.5 * inv_a + torch.square(z_middle) * (2.0 * torch.pow(inv_a, 3)))
            result[middle_mask] = middle_result
        
        if lower_mask.any():
            result[lower_mask] = 0.0
        
        return result


@register_activation
class Elliott(nn.Module):
    r"""
    Applies the Elliott Activation Function:

    :math:`\text{Elliott}(z) = \frac{0.5z}{1 + |z|} + 0.5`
    :note: Elliott should be faster than Sigmoid

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Elliott()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(Elliott, self).__init__()

    def forward(self, z) -> Tensor:
        return (0.5 * z) / (1 + torch.abs(z)) + 0.5


@register_activation
class SincSigmoid(nn.Module):
    r"""
    Applies the Sinc Sigmoid activation function:

    :math:`\text{SincSigmoid}(z) = \text{sinc}(\sigma(z))`

    where :math:`\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}` if :math:`x \neq 0`, and 1 if :math:`x = 0`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SincSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SincSigmoid, self).__init__()

    def forward(self, z) -> Tensor:
        sigmoid_z = torch.sigmoid(z)
        
        result = torch.ones_like(z)
        nonzero_mask = sigmoid_z > 1e-10

        # Sinc is defined as sin(pi * x) / (pi * x)
        pi_sigmoid_z = torch.pi * sigmoid_z[nonzero_mask]
        result[nonzero_mask] = torch.sin(pi_sigmoid_z) / pi_sigmoid_z

        return result


@register_activation
class SigmoidGumbel(nn.Module):
    r"""
    Applies the Sigmoid Gumbel activation function:

    :math:`\text{SigmoidGumbel}(z) = \frac{1}{1 + \exp(-z) \exp(-\exp(-z))}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SigmoidGumbel()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SigmoidGumbel, self).__init__()

    def forward(self, z) -> Tensor:
        neg_z = -z
        # Sigmoid is defined as 1 / (1 + exp(-z))
        return torch.sigmoid(z) * torch.exp(-torch.exp(neg_z))


@register_activation
class NewSigmoid(nn.Module):
    r"""
    Applies the New Sigmoid activation function:

    :math:`\text{NewSigmoid}(z) = \frac{\exp(z) - \exp(-z)}{2(\exp(2z) + \exp(-2z))}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.NewSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(NewSigmoid, self).__init__()

    def forward(self, z) -> Tensor:
        exp_z = torch.exp(z)
        exp_neg_z = torch.exp(-z)
        exp_2z = torch.exp(2 * z)
        exp_neg_2z = torch.exp(-2 * z)

        numerator = exp_z - exp_neg_z
        denominator = torch.sqrt(2 * (exp_2z + exp_neg_2z))

        return numerator / denominator


@register_activation
class Root2sigmoid(nn.Module):
    r"""
    Applies the Root2sigmoid activation function:

    :math:`\text{Root2sigmoid}(z) = \frac{\sqrt{2}^z - \sqrt{2}^{-z}}{2 \cdot \sqrt{2} \cdot \sqrt{2 \cdot (\sqrt{2}^{2z} + \sqrt{2}^{-2z})}}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Root2sigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(Root2sigmoid, self).__init__()
        self.r = torch.sqrt(torch.tensor(2.0))

    def forward(self, z) -> Tensor:
        numerator = torch.pow(self.r, z) - torch.pow(self.r, -z)
        denominator = 2 * self.r * (torch.sqrt(2 * (torch.pow(self.r, 2 * z) + torch.pow(self.r, -2 * z))))
        return numerator / denominator


@register_activation
class LogLog(nn.Module):
    r"""
    Applies the LogLog activation function:

    :math:`\text{LogLog}(z) = \exp(-\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.LogLog()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(LogLog, self).__init__()

    def forward(self, z) -> Tensor:
        return torch.exp(-torch.exp(-z))


@register_activation
class cLogLog(nn.Module):
    r"""
    Applies the Complementary LogLog activation function:

    :math:`\text{cLogLog}(z) = 1 - \exp(-\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.cLogLog()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(cLogLog, self).__init__()

    def forward(self, z) -> Tensor:
        return 1 - torch.exp(-torch.exp(-z))


@register_activation
class cLogLogm(nn.Module):
    r"""
    Applies the Modified Complementary LogLog activation function:

    :math:`\text{cLogLogm}(z) = 1 - 2\exp(-0.7\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.cLogLogm()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(cLogLogm, self).__init__()

    def forward(self, z) -> Tensor:
        return 1 - 2 * torch.exp(-0.7 * torch.exp(-z))


@register_activation
class SechSig(nn.Module):
    r"""
    Applies the SechSig activation function:

    :math:`\text{SechSig}(z) = (z + \text{sech}(z))\sigma(z)`

    where :math:`\text{sech}(z) = \frac{2}{e^z + e^{-z}}` is the hyperbolic secant.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SechSig()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SechSig, self).__init__()

    def forward(self, z) -> Tensor:
        return (z + sech(z)) * torch.sigmoid(z)


@register_activation
class pSechSig(nn.Module):
    r"""
    Applies the Parametric SechSig activation function:

    :math:`\text{pSechSig}(z) = (z + a\cdot \text{sech}(z+a))\sigma(z)`

    where :math:`\text{sech}(z) = \frac{2}{e^z + e^{-z}}` is the hyperbolic secant.

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.pSechSig(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(pSechSig, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

    def forward(self, z) -> Tensor:
        return (z + self.a * sech(z + self.a)) * torch.sigmoid(z)


@register_activation
class TanhSig(nn.Module):
    r"""
    Applies the TanhSig activation function:

    :math:`\text{TanhSig}(z) = (z + \tanh(z))\sigma(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.TanhSig()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(TanhSig, self).__init__()

    def forward(self, z) -> Tensor:
        return (z + torch.tanh(z)) * torch.sigmoid(z)


@register_activation
class pTanhSig(nn.Module):
    r"""
    Applies the Parametric TanhSig activation function:

    :math:`\text{pTanhSig}(z) = (z + a\cdot \tanh(z+a))\sigma(z)`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.pTanhSig(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(pTanhSig, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

    def forward(self, z) -> Tensor:
        return (z + self.a * torch.tanh(z + self.a)) * torch.sigmoid(z)


@register_activation
class MSAF(nn.Module):
    r"""
    Applies the Multistate Activation Function:

    :math:`\text{MSAF}(z) = a + \sum_{k=1}^N \frac{1}{1 + \exp(-z+b_k)}`

    Args:
        a (float, optional): Offset parameter. Default: 0.0
        b (list of float, optional): List of shift parameters. Default: [1.0, 2.0]

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.MSAF(a=0.5, b=[0.5, 1.5, 2.5])
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.0, b: list = None):
        super(MSAF, self).__init__()
        if b is None:
            b = [1.0, 2.0]
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)

    def forward(self, z) -> Tensor:
        result = self.a.expand_as(z)

        for k in range(len(self.b)):
            result = result + torch.sigmoid(z - self.b[k])

        return result


@register_activation
class SymMSAF(nn.Module):
    r"""
    Applies the Symmetrical Multistate Activation Function:

    :math:`\text{SymMSAF}(z) = -1 + \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z-a)}`

    Args:
        a (float, optional): Shift parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SymMSAF(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(SymMSAF, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        return -1 + torch.sigmoid(z) + torch.sigmoid(z + self.a)


@register_activation
class Rootsig(nn.Module):
    r"""
    Applies the Rootsig activation function:

    :math:`\text{Rootsig}(z) = \frac{az}{1 + \sqrt{1 + a^2z^2}}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.Rootsig(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(Rootsig, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

    def forward(self, z) -> Tensor:
        a_z = self.a * z
        return a_z / (1 + torch.sqrt(1 + a_z * a_z))


# The math is equivalent to RootsigPlus
# @register_activation
# class UnnamedSigmoid1(nn.Module):
#     # TODO: Ask someone about this name. 
#     r"""
#     :note: The name "UnnamedSigmoid1" derived from the first entry in "3.2.25 Rootsig and others" entry. I named it this way because the curve resembles the Rootsig but not as soft

#     Applies the RootsigPlus activation function:

#     :math:`\text{UnnamedSigmoid1}(z) = z \cdot \text{sgn}(z) \sqrt{z^{-a} - a^{-2}}`

#     Args:
#         a (float, optional): Shape parameter. Default: 1.0

#     Shape:
#         - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
#         - Output: :math:`(*)`, same shape as the input.

#     Examples::

#         >>> m = tac.UnnamedSigmoid1(a=3.0)
#         >>> x = torch.randn(2)
#         >>> output = m(x)
#     """

#     def __init__(self, a: float = 1.0):
#         super(UnnamedSigmoid1, self).__init__()
#         self.a = nn.Parameter(torch.tensor([a]), requires_grad=False)

#     def forward(self, z) -> Tensor:
#         numerator = torch.sign(z) * z - self.a
#         denominator = torch.square(z) - torch.square(self.a)
#         return z * (numerator / denominator)


@register_activation
class RootsigPlus(nn.Module):
    # TODO: Ask someone about this name. 
    r"""
    :note: The name "RootsigPlus" derived from the second entry in "3.2.25 Rootsig and others" entry, found in the `Estimates of the number of hidden units and variation with respect
    to half-spaces` paper. I named it this way because the curve resembles the Tanh but softer and not as soft as Rootsig.
    Applies the Radical Tanh activation function:

    :math:`\text{RootsigPlus}(z) = \frac{az}{1 + |az|}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.RootsigPlus(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(RootsigPlus, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        a_z = self.a * z
        return a_z / (1 + torch.abs(a_z))



@register_activation
class SoftTanh(nn.Module):
    # TODO: Ask someone about this name. 
    r"""
    :note: The name "RadicalTanh" derived from the third entry in "3.2.25 Rootsig and others" entry, found in the `Estimates of the number of hidden units and variation with respect
    to half-spaces` paper. I named it this way because the curve resembles the Tanh but softer and not as soft as RootsigPlus.
    Applies the SoftTanh activation function:

    :math:`\text{SoftTanh}(z) = \frac{az}{\sqrt{1 + a^2z^2}}`

    Args:
        a (float, optional): Shape parameter. Default: 2.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SoftTanh(a=3.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 2.0):
        super(SoftTanh, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        a_z = self.a * z
        return a_z / (torch.sqrt(1 + a_z * a_z))


@register_activation
class SigmoidTanh(nn.Module):
    r"""
    Applies the Sigmoid-Tanh Combinations activation function:

    :math:`\text{SigmoidTanh}(z) = \begin{cases} 
    g(z), & z \geq 0 \\ 
    h(z), & z < 0 
    \end{cases}`

    where g(z) and h(z) are user-defined functions, defaulting to sigmoid and tanh respectively.

    Args:
        g_func (callable, optional): Function to use for positive inputs. Default: torch.sigmoid
        h_func (callable, optional): Function to use for negative inputs. Default: torch.tanh

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tac.SigmoidTanh()
        >>> x = torch.randn(2)
        >>> output = m(x)
        
        >>> # Custom functions
        >>> import torch.nn.functional as F
        >>> m = tac.SigmoidTanh(g_func=F.relu, h_func=torch.sigmoid)
        >>> output = m(x)
    """

    def __init__(self, g_func=torch.sigmoid, h_func=torch.tanh):
        super(SigmoidTanh, self).__init__()
        self.g_func = g_func
        self.h_func = h_func

    def forward(self, z) -> Tensor:
        pos_mask = z >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(z)
        result[pos_mask] = self.g_func(z[pos_mask])
        result[neg_mask] = self.h_func(z[neg_mask])

        return result
