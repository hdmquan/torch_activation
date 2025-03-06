import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch_activation import register_activation

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

        >>> m = nn.Sigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.Sigmoid(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()

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

        >>> m = nn.Tanh()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.Tanh(inplace=True)
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

        >>> m = nn.ShiftedScaledSigmoid(a=2.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.ShiftedScaledSigmoid(inplace=True)
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

        >>> m = nn.VariantSigmoidFunction(a=2.0, b=1.5, c=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.VariantSigmoidFunction(inplace=True)
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

    Args:
        a (float, optional): Scale parameter. Default: 1.0
        b (float, optional): Slope parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.STanh(a=1.7, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.STanh(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, inplace: bool = False):
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

#         >>> m = nn.BiModalDerivativeSigmoid(b=2.0)
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

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Arctan()
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

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ArctanGR()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ArctanGR, self).__init__()
        self.scale_factor = 1.0 / (1.0 + torch.sqrt(torch.tensor(2.0)))

    def forward(self, z) -> Tensor:
        return torch.atan(z) * self.scale_factor

@register_activation
class SigmoidAlgebraic(nn.Module):
    r"""
    Applies the Sigmoid Algebraic activation function:

    :math:`\text{SigmoidAlgebraic}(z) = \frac{1}{1 + \exp\left(-\frac{z(1 + a|z|)}{1 + |z|(1 + a|z|)}\right)}`

    Args:
        a (float, optional): Shape parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SigmoidAlgebraic(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SigmoidAlgebraic(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(SigmoidAlgebraic, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        abs_z = torch.abs(z)
        a_abs_z = self.a * abs_z
        numerator = z * (1 + a_abs_z)
        denominator = 1 + abs_z * (1 + a_abs_z)
        return torch.sigmoid(numerator / denominator)

    def _forward_inplace(self, z):
        # Cannot be done fully in-place due to the complex calculation
        abs_z = torch.abs(z)
        a_abs_z = self.a * abs_z
        numerator = z.clone() * (1 + a_abs_z)
        denominator = 1 + abs_z * (1 + a_abs_z)
        z.copy_(numerator / denominator).sigmoid_()
        return z


@register_activation
class TripleStateSigmoid(nn.Module):
    r"""
    Applies the Triple State Sigmoid activation function:

    :math:`\text{TripleStateSigmoid}(z) = \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z+a)} + \frac{1}{1 + \exp(-z+b)}`

    Args:
        a (float, optional): First shift parameter. Default: 1.0
        b (float, optional): Second shift parameter. Default: 2.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.TripleStateSigmoid(a=1.5, b=2.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.TripleStateSigmoid(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, inplace: bool = False):
        super(TripleStateSigmoid, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return torch.sigmoid(z) + torch.sigmoid(z - self.a) + torch.sigmoid(z - self.b)

    def _forward_inplace(self, z):
        # Cannot be done fully in-place due to the need for the original z value
        result = torch.sigmoid(z.clone())
        result.add_(torch.sigmoid(z - self.a))
        result.add_(torch.sigmoid(z - self.b))
        z.copy_(result)
        return z


@register_activation
class ImprovedLogisticSigmoid(nn.Module):
    r"""
    Applies the Improved Logistic Sigmoid activation function:

    :math:`\text{ImprovedLogisticSigmoid}(z) = \begin{cases} 
    a(z-b) + \sigma(b), & z \geq b \\ 
    \sigma(z), & -b < z < b \\ 
    a(z+b) + \sigma(-b), & z \leq -b 
    \end{cases}`

    Args:
        a (float, optional): Slope parameter. Default: 0.2
        b (float, optional): Threshold parameter. Default: 2.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ImprovedLogisticSigmoid(a=0.1, b=3.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.ImprovedLogisticSigmoid(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.2, b: float = 2.0, inplace: bool = False):
        super(ImprovedLogisticSigmoid, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        sig_b = torch.sigmoid(self.b)
        sig_neg_b = torch.sigmoid(-self.b)

        upper_region = self.a * (z - self.b) + sig_b
        middle_region = torch.sigmoid(z)
        lower_region = self.a * (z + self.b) + sig_neg_b

        result = torch.where(z >= self.b, upper_region, middle_region)
        result = torch.where(z <= -self.b, lower_region, result)

        return result

    def _forward_inplace(self, z):
        # Cannot be done fully in-place due to the conditional nature
        sig_b = torch.sigmoid(self.b)
        sig_neg_b = torch.sigmoid(-self.b)

        upper_mask = z >= self.b
        lower_mask = z <= -self.b
        middle_mask = ~(upper_mask | lower_mask)

        result = z.clone()

        # Apply each region's transformation
        result[upper_mask] = self.a * (z[upper_mask] - self.b) + sig_b
        result[middle_mask] = torch.sigmoid(z[middle_mask])
        result[lower_mask] = self.a * (z[lower_mask] + self.b) + sig_neg_b

        z.copy_(result)
        return z


@register_activation
class SigLin(nn.Module):
    r"""
    Applies the SigLin activation function:

    :math:`\text{SigLin}(z) = \sigma(z) + az`

    Args:
        a (float, optional): Linear coefficient. Default: 0.2
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SigLin(a=0.1)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SigLin(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 0.2, inplace: bool = False):
        super(SigLin, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return torch.sigmoid(z) + self.a * z

    def _forward_inplace(self, z):
        # Cannot be done fully in-place
        z_copy = z.clone()
        z.sigmoid_().add_(self.a * z_copy)
        return z


@register_activation
class PenalizedHyperbolicTangent(nn.Module):
    r"""
    Applies the Penalized Hyperbolic Tangent activation function:

    :math:`\text{PenalizedHyperbolicTangent}(z) = \begin{cases} 
    \tanh(z), & z \geq 0 \\ 
    \frac{\tanh(z)}{a}, & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Penalty factor for negative inputs. Default: 2.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.PenalizedHyperbolicTangent(a=3.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.PenalizedHyperbolicTangent(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 2.0, inplace: bool = False):
        super(PenalizedHyperbolicTangent, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        tanh_z = torch.tanh(z)
        return torch.where(z >= 0, tanh_z, tanh_z / self.a)

    def _forward_inplace(self, z):
        # Cannot be done fully in-place due to the conditional nature
        neg_mask = z < 0
        z.tanh_()
        z[neg_mask].div_(self.a)
        return z


@register_activation
class SoftRootSign(nn.Module):
    r"""
    Applies the Soft Root Sign activation function:

    :math:`\text{SoftRootSign}(z) = \frac{z}{\sqrt[a]{1 + \exp\left(-\frac{z}{b}\right)}}`

    Args:
        a (float, optional): Root parameter. Default: 2.0
        b (float, optional): Scale parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SoftRootSign(a=3.0, b=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SoftRootSign(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 2.0, b: float = 1.0, inplace: bool = False):
        super(SoftRootSign, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        denominator = torch.pow(1 + torch.exp(-z / self.b), 1 / self.a)
        return z / denominator

    def _forward_inplace(self, z):
        # Cannot be done fully in-place
        z_copy = z.clone()
        denominator = torch.pow(1 + torch.exp(-z / self.b), 1 / self.a)
        z.copy_(z_copy / denominator)
        return z


@register_activation
class SoftClipping(nn.Module):
    r"""
    Applies the Soft Clipping activation function:

    :math:`\text{SoftClipping}(z) = \frac{1}{a} \ln\left(\frac{1 + \exp(az)}{1 + \exp(a(z-1))}\right)`

    Args:
        a (float, optional): Sharpness parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SoftClipping(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SoftClipping(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(SoftClipping, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        numerator = 1 + torch.exp(self.a * z)
        denominator = 1 + torch.exp(self.a * (z - 1))
        return torch.log(numerator / denominator) / self.a

    def _forward_inplace(self, z):
        # Cannot be done fully in-place
        numerator = 1 + torch.exp(self.a * z)
        denominator = 1 + torch.exp(self.a * (z - 1))
        z.copy_(torch.log(numerator / denominator) / self.a)
        return z


@register_activation
class Hexpo(nn.Module):
    r"""
    Applies the Hexpo activation function:

    :math:`\text{Hexpo}(z) = \begin{cases} 
    -a \exp\left(-\frac{z}{b}\right) - 1, & z \geq 0 \\ 
    c \exp\left(-\frac{z}{d}\right) - 1, & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Positive scale parameter. Default: 1.0
        b (float, optional): Positive decay parameter. Default: 1.0
        c (float, optional): Negative scale parameter. Default: 1.0
        d (float, optional): Negative decay parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Hexpo(a=1.5, b=0.5, c=2.0, d=0.7)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.Hexpo(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        inplace: bool = False,
    ):
        super(Hexpo, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.c = nn.Parameter(torch.tensor([c]))
        self.d = nn.Parameter(torch.tensor([d]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        pos_part = -self.a * torch.exp(-z / self.b) - 1
        neg_part = self.c * torch.exp(-z / self.d) - 1
        return torch.where(z >= 0, pos_part, neg_part)

    def _forward_inplace(self, z):
        # Cannot be done fully in-place due to the conditional nature
        pos_mask = z >= 0
        neg_mask = ~pos_mask

        result = z.clone()
        result[pos_mask] = -self.a * torch.exp(-z[pos_mask] / self.b) - 1
        result[neg_mask] = self.c * torch.exp(-z[neg_mask] / self.d) - 1

        z.copy_(result)
        return z


@register_activation
class Softsign(nn.Module):
    r"""
    Applies the Softsign activation function:

    :math:`\text{Softsign}(z) = \frac{z}{1 + |z|}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Softsign()
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
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SmoothStep(a=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SmoothStep(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 2.0, inplace: bool = False):
        super(SmoothStep, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        half_a = self.a / 2

        # Calculate the polynomial for the middle region
        cubic_term = (2 / (self.a**3)) * (z**3)
        linear_term = (3 / (2 * self.a)) * z
        constant_term = 0.5
        middle_region = cubic_term - linear_term + constant_term

        # Apply the piecewise function
        result = torch.ones_like(z)
        middle_mask = (z > -half_a) & (z < half_a)
        lower_mask = z <= -half_a

        result[middle_mask] = middle_region[middle_mask]
        result[lower_mask] = 0.0

        return result

    def _forward_inplace(self, z):
        # Cannot be done fully in-place due to the conditional nature
        half_a = self.a / 2

        # Create masks for each region
        upper_mask = z >= half_a
        middle_mask = (z > -half_a) & (z < half_a)
        lower_mask = z <= -half_a

        # Calculate the polynomial for the middle region
        middle_values = z[middle_mask]
        cubic_term = (2 / (self.a**3)) * (middle_values**3)
        linear_term = (3 / (2 * self.a)) * middle_values
        middle_result = cubic_term - linear_term + 0.5

        # Apply the piecewise function
        result = z.clone()
        result[upper_mask] = 1.0
        result[middle_mask] = middle_result
        result[lower_mask] = 0.0

        z.copy_(result)
        return z


@register_activation
class ElliottActivationFunction(nn.Module):
    r"""
    Applies the Elliott Activation Function:

    :math:`\text{ElliottActivationFunction}(z) = \frac{0.5z}{1 + |z|} + 0.5`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ElliottActivationFunction()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ElliottActivationFunction, self).__init__()

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

        >>> m = nn.SincSigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SincSigmoid, self).__init__()

    def forward(self, z) -> Tensor:
        sigmoid_z = torch.sigmoid(z)
        # Handle the case where sigmoid_z is 0 to avoid division by zero
        result = torch.ones_like(z)
        nonzero_mask = sigmoid_z != 0

        # Calculate sinc for non-zero values
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

        >>> m = nn.SigmoidGumbel()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SigmoidGumbel, self).__init__()

    def forward(self, z) -> Tensor:
        neg_z = -z
        return 1 / (1 + torch.exp(neg_z) * torch.exp(-torch.exp(neg_z)))

@register_activation
class NewSigmoid(nn.Module):
    r"""
    Applies the New Sigmoid activation function:

    :math:`\text{NewSigmoid}(z) = \frac{\exp(z) - \exp(-z)}{2(\exp(2z) + \exp(-2z))}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.NewSigmoid()
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
        denominator = 2 * (exp_2z + exp_neg_2z)

        return numerator / denominator


@register_activation
class Root2sigmoid(nn.Module):
    r"""
    Applies the Root2sigmoid activation function:

    :math:`\text{Root2sigmoid}(z) = \frac{\sqrt{2}z}{\sqrt{2^{-2z}} + \sqrt{2^{2z}}}`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Root2sigmoid()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(Root2sigmoid, self).__init__()
        self.sqrt2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, z) -> Tensor:
        sqrt2_z = self.sqrt2 * z
        neg_2z = -2 * z
        pos_2z = 2 * z

        denominator = torch.sqrt(torch.pow(2, neg_2z)) + torch.sqrt(
            torch.pow(2, pos_2z)
        )
        return sqrt2_z / denominator


@register_activation
class LogLog(nn.Module):
    r"""
    Applies the LogLog activation function:

    :math:`\text{LogLog}(z) = \exp(-\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.LogLog()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(LogLog, self).__init__()

    def forward(self, z) -> Tensor:
        return torch.exp(-torch.exp(-z))


@register_activation
class ComplementaryLogLog(nn.Module):
    r"""
    Applies the Complementary LogLog activation function:

    :math:`\text{ComplementaryLogLog}(z) = 1 - \exp(-\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ComplementaryLogLog()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ComplementaryLogLog, self).__init__()

    def forward(self, z) -> Tensor:
        return 1 - torch.exp(-torch.exp(-z))


@register_activation
class ModifiedComplementaryLogLog(nn.Module):
    r"""
    Applies the Modified Complementary LogLog activation function:

    :math:`\text{ModifiedComplementaryLogLog}(z) = 1 - 2\exp(-0.7\exp(-z))`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ModifiedComplementaryLogLog()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(ModifiedComplementaryLogLog, self).__init__()

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

        >>> m = nn.SechSig()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(SechSig, self).__init__()

    def forward(self, z) -> Tensor:
        # Calculate hyperbolic secant: sech(z) = 2/(e^z + e^-z)
        exp_z = torch.exp(z)
        exp_neg_z = torch.exp(-z)
        sech_z = 2 / (exp_z + exp_neg_z)

        return (z + sech_z) * torch.sigmoid(z)


@register_activation
class ParametricSechSig(nn.Module):
    r"""
    Applies the Parametric SechSig activation function:

    :math:`\text{ParametricSechSig}(z) = (z + a\cdot \text{sech}(z+a))\sigma(z)`

    where :math:`\text{sech}(z) = \frac{2}{e^z + e^{-z}}` is the hyperbolic secant.

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ParametricSechSig(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(ParametricSechSig, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        # Calculate shifted input
        z_plus_a = z + self.a

        # Calculate hyperbolic secant: sech(z+a) = 2/(e^(z+a) + e^-(z+a))
        exp_z_plus_a = torch.exp(z_plus_a)
        exp_neg_z_plus_a = torch.exp(-z_plus_a)
        sech_z_plus_a = 2 / (exp_z_plus_a + exp_neg_z_plus_a)

        return (z + self.a * sech_z_plus_a) * torch.sigmoid(z)


@register_activation
class TanhSig(nn.Module):
    r"""
    Applies the TanhSig activation function:

    :math:`\text{TanhSig}(z) = (z + \tanh(z))\sigma(z)`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.TanhSig()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        super(TanhSig, self).__init__()

    def forward(self, z) -> Tensor:
        return (z + torch.tanh(z)) * torch.sigmoid(z)


@register_activation
class ParametricTanhSig(nn.Module):
    r"""
    Applies the Parametric TanhSig activation function:

    :math:`\text{ParametricTanhSig}(z) = (z + a\cdot \tanh(z+a))\sigma(z)`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ParametricTanhSig(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(ParametricTanhSig, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        return (z + self.a * torch.tanh(z + self.a)) * torch.sigmoid(z)


@register_activation
class MultistateActivationFunction(nn.Module):
    r"""
    Applies the Multistate Activation Function:

    :math:`\text{MultistateActivationFunction}(z) = a + \sum_{k=1}^N \frac{1}{1 + \exp(-z+b_k)}`

    Args:
        a (float, optional): Offset parameter. Default: 0.0
        b (list of float, optional): List of shift parameters. Default: [1.0, 2.0]

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.MultistateActivationFunction(a=0.5, b=[0.5, 1.5, 2.5])
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 0.0, b: list = None):
        super(MultistateActivationFunction, self).__init__()
        if b is None:
            b = [1.0, 2.0]
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, z) -> Tensor:
        result = self.a.clone()

        for k in range(len(self.b)):
            result = result + torch.sigmoid(z - self.b[k])

        return result


@register_activation
class SymmetricalMSAF(nn.Module):
    r"""
    Applies the Symmetrical Multistate Activation Function:

    :math:`\text{SymmetricalMSAF}(z) = -1 + \frac{1}{1 + \exp(-z)} + \frac{1}{1 + \exp(-z-a)}`

    Args:
        a (float, optional): Shift parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SymmetricalMSAF(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(SymmetricalMSAF, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        return -1 + torch.sigmoid(z) + torch.sigmoid(z - self.a)


@register_activation
class Rootsig(nn.Module):
    r"""
    Applies the Rootsig activation function:

    :math:`\text{Rootsig}(z) = \frac{az}{\sqrt{1 + a^2z^2}}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Rootsig(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(Rootsig, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        a_z = self.a * z
        return a_z / torch.sqrt(1 + a_z * a_z)


@register_activation
class UnnamedSigmoid1(nn.Module):
    # TODO: Naming
    r"""
    :note: The name "UnnamesSigmoid1" derived from "3.2.25 Rootsig and others" entry, particularly the first from from "others" part. Leave it here until I find a better name.
    Applies the Unnamed Sigmoid 1 activation function:

    :math:`\text{UnnamedSigmoid1}(z) = z \cdot \text{sgn}(z) \sqrt{z^{-a} - a^{-2}}`

    Args:
        a (float, optional): Shape parameter. Default: 2.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.UnnamedSigmoid1(a=3.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 2.0):
        super(UnnamedSigmoid1, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.a_squared_inv = 1 / (a * a)

    def forward(self, z) -> Tensor:
        # Handle the sign of z
        sign_z = torch.sign(z)
        abs_z = torch.abs(z)

        # Avoid division by zero or negative values under sqrt
        eps = 1e-6
        safe_z = torch.clamp(abs_z, min=eps)

        # Calculate z^(-a)
        z_pow_neg_a = torch.pow(safe_z, -self.a)

        # Calculate sqrt term, ensuring the value under sqrt is positive
        sqrt_term = torch.sqrt(torch.clamp(z_pow_neg_a - self.a_squared_inv, min=0))

        return z * sign_z * sqrt_term


@register_activation
class UnnamedSigmoid2(nn.Module):
    # TODO: Naming
    r"""
    :note: The name "UnnamesSigmoid2" derived from "3.2.25 Rootsig and others" entry, particularly the second from from "others" part. Leave it here until I find a better name.
    Applies the Unnamed Sigmoid 2 activation function:

    :math:`\text{UnnamedSigmoid2}(z) = \frac{az}{1 + |az|}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.UnnamedSigmoid2(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(UnnamedSigmoid2, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        a_z = self.a * z
        return a_z / (1 + torch.abs(a_z))


@register_activation
class UnnamedSigmoid3(nn.Module):
    # TODO: Naming
    r"""
    :note: The name "UnnamedSigmoid3" derived from "3.2.25 Rootsig and others" entry, particularly the third from from "others" part. Leave it here until I find a better name.
    Applies the Unnamed Sigmoid 3 activation function:

    :math:`\text{UnnamedSigmoid3}(z) = \frac{az}{\sqrt{1 + a^2z^2}}`

    Args:
        a (float, optional): Scale parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.UnnamedSigmoid3(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0):
        super(UnnamedSigmoid3, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))

    def forward(self, z) -> Tensor:
        a_z = self.a * z
        return a_z / torch.sqrt(1 + a_z * a_z)


@register_activation
class SigmoidTanhCombinations(nn.Module):
    r"""
    Applies the Sigmoid-Tanh Combinations activation function:

    :math:`\text{SigmoidTanhCombinations}(z) = \begin{cases} 
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

        >>> m = nn.SigmoidTanhCombinations()
        >>> x = torch.randn(2)
        >>> output = m(x)
        
        >>> # Custom functions
        >>> import torch.nn.functional as F
        >>> m = nn.SigmoidTanhCombinations(g_func=F.relu, h_func=torch.sigmoid)
        >>> output = m(x)
    """

    def __init__(self, g_func=torch.sigmoid, h_func=torch.tanh):
        super(SigmoidTanhCombinations, self).__init__()
        self.g_func = g_func
        self.h_func = h_func

    def forward(self, z) -> Tensor:
        pos_mask = z >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(z)
        result[pos_mask] = self.g_func(z[pos_mask])
        result[neg_mask] = self.h_func(z[neg_mask])

        return result
