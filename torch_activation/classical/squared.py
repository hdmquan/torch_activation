import torch
import torch.nn as nn
from torch import Tensor
from torch_activation import register_activation


@register_activation
class SQNL(nn.Module):
    r"""
    Applies the SQNL (Square Non-Linear) activation function:

    :math:`\text{SQNL}(z) = \begin{cases} 
    1, & z > 2 \\ 
    z - \frac{z^2}{4}, & 0 \leq z \leq 2 \\ 
    z + \frac{z^2}{4}, & -2 \leq z < 0 \\ 
    -1, & z < -2 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SQNL()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SQNL(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(SQNL, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        gt2 = z > 2
        between0and2 = (z >= 0) & (z <= 2)
        betweenNeg2and0 = (z >= -2) & (z < 0)
        ltNeg2 = z < -2
        
        result[gt2] = 1
        result[between0and2] = z[between0and2] - (z[between0and2] ** 2) / 4
        result[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 4
        result[ltNeg2] = -1
        
        return result

    def _forward_inplace(self, z):
        gt2 = z > 2
        between0and2 = (z >= 0) & (z <= 2)
        betweenNeg2and0 = (z >= -2) & (z < 0)
        ltNeg2 = z < -2
        
        z[gt2] = 1
        z[between0and2] = z[between0and2] - (z[between0and2] ** 2) / 4
        z[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 4
        z[ltNeg2] = -1
        
        return z


@register_activation
class SQLU(nn.Module):
    r"""
    Applies the SQLU (Square Linear Unit) activation function:

    :math:`\text{SQLU}(z) = \begin{cases} 
    z, & z > 0 \\ 
    z + \frac{z^2}{4}, & -2 \leq z \leq 0 \\ 
    -1, & z < -2 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SQLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SQLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(SQLU, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        gt0 = z > 0
        betweenNeg2and0 = (z >= -2) & (z <= 0)
        ltNeg2 = z < -2
        
        result[gt0] = z[gt0]
        result[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 4
        result[ltNeg2] = -1
        
        return result

    def _forward_inplace(self, z):
        gt0 = z > 0
        betweenNeg2and0 = (z >= -2) & (z <= 0)
        ltNeg2 = z < -2
        
        # No change needed for z > 0
        z[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 4
        z[ltNeg2] = -1
        
        return z


@register_activation
class Squish(nn.Module):
    r"""
    Applies the Squish activation function:

    :math:`\text{Squish}(z) = \begin{cases} 
    z + \frac{z^2}{32}, & z > 0 \\ 
    z + \frac{z^2}{2}, & -2 \leq z \leq 0 \\ 
    0, & z < -2 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Squish()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.Squish(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(Squish, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        gt0 = z > 0
        betweenNeg2and0 = (z >= -2) & (z <= 0)
        ltNeg2 = z < -2
        
        result[gt0] = z[gt0] + (z[gt0] ** 2) / 32
        result[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 2
        result[ltNeg2] = 0
        
        return result

    def _forward_inplace(self, z):
        gt0 = z > 0
        betweenNeg2and0 = (z >= -2) & (z <= 0)
        ltNeg2 = z < -2
        
        z[gt0] = z[gt0] + (z[gt0] ** 2) / 32
        z[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 2
        z[ltNeg2] = 0
        
        return z


@register_activation
class SqREU(nn.Module):
    r"""
    Applies the SqREU (Square Rectified Exponential Unit) activation function:

    :math:`\text{SqREU}(z) = \begin{cases} 
    z, & z > 0 \\ 
    z + \frac{z^2}{2}, & -2 \leq z \leq 0 \\ 
    0, & z < -2 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SqREU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SqREU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(SqREU, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        gt0 = z > 0
        betweenNeg2and0 = (z >= -2) & (z <= 0)
        ltNeg2 = z < -2
        
        result[gt0] = z[gt0]
        result[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 2
        result[ltNeg2] = 0
        
        return result

    def _forward_inplace(self, z):
        gt0 = z > 0
        betweenNeg2and0 = (z >= -2) & (z <= 0)
        ltNeg2 = z < -2
        
        # No change needed for z > 0
        z[betweenNeg2and0] = z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 2
        z[ltNeg2] = 0
        
        return z


@register_activation
class SqSoftplus(nn.Module):
    r"""
    Applies the SqSoftplus (Square Softplus) activation function:

    :math:`\text{SqSoftplus}(z) = \begin{cases} 
    z, & z > \frac{1}{2} \\ 
    z + \frac{(z + \frac{1}{2})^2}{2}, & -\frac{1}{2} \leq z \leq \frac{1}{2} \\ 
    0, & z < -\frac{1}{2} 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SqSoftplus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SqSoftplus(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(SqSoftplus, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        gtHalf = z > 0.5
        betweenNegHalfAndHalf = (z >= -0.5) & (z <= 0.5)
        ltNegHalf = z < -0.5
        
        result[gtHalf] = z[gtHalf]
        result[betweenNegHalfAndHalf] = z[betweenNegHalfAndHalf] + ((z[betweenNegHalfAndHalf] + 0.5) ** 2) / 2
        result[ltNegHalf] = 0
        
        return result

    def _forward_inplace(self, z):
        gtHalf = z > 0.5
        betweenNegHalfAndHalf = (z >= -0.5) & (z <= 0.5)
        ltNegHalf = z < -0.5
        
        # No change needed for z > 0.5
        z[betweenNegHalfAndHalf] = z[betweenNegHalfAndHalf] + ((z[betweenNegHalfAndHalf] + 0.5) ** 2) / 2
        z[ltNegHalf] = 0
        
        return z


@register_activation
class LogSQNL(nn.Module):
    r"""
    Applies the LogSQNL (Logarithmic Square Non-Linear) activation function:

    :math:`\text{LogSQNL}(z) = \begin{cases} 
    1, & z > 2 \\ 
    \frac{1}{2}z - \frac{z^2}{4} + \frac{1}{2}, & 0 \leq z \leq 2 \\ 
    \frac{1}{2}z + \frac{z^2}{4} + \frac{1}{2}, & -2 \leq z < 0 \\ 
    0, & z < -2 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.LogSQNL()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.LogSQNL(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(LogSQNL, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        gt2 = z > 2
        between0and2 = (z >= 0) & (z <= 2)
        betweenNeg2and0 = (z >= -2) & (z < 0)
        ltNeg2 = z < -2
        
        result[gt2] = 1
        result[between0and2] = 0.5 * z[between0and2] - (z[between0and2] ** 2) / 4 + 0.5
        result[betweenNeg2and0] = 0.5 * z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 4 + 0.5
        result[ltNeg2] = 0
        
        return result

    def _forward_inplace(self, z):
        gt2 = z > 2
        between0and2 = (z >= 0) & (z <= 2)
        betweenNeg2and0 = (z >= -2) & (z < 0)
        ltNeg2 = z < -2
        
        z[gt2] = 1
        z[between0and2] = 0.5 * z[between0and2] - (z[between0and2] ** 2) / 4 + 0.5
        z[betweenNeg2and0] = 0.5 * z[betweenNeg2and0] + (z[betweenNeg2and0] ** 2) / 4 + 0.5
        z[ltNeg2] = 0
        
        return z


@register_activation
class SQMAX(nn.Module):
    r"""
    Applies the SQMAX (Square Maximum) activation function:

    :math:`\text{SQMAX}(z_j) = \frac{(z_j + c)^2}{\sum_{k=1}^N (z_k + c)^2}`

    Args:
        c (float, optional): Offset parameter. Default: 0.0
        dim (int, optional): A dimension along which SQMAX will be computed. Default: -1

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SQMAX(c=1.0)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = nn.SQMAX(dim=0)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, c: float = 0.0, dim: int = -1):
        super(SQMAX, self).__init__()
        self.c = nn.Parameter(torch.tensor([c]))
        self.dim = dim

    def forward(self, z) -> Tensor:
        shifted = z + self.c
        squared = shifted ** 2
        sum_squared = squared.sum(dim=self.dim, keepdim=True)
        return squared / sum_squared


@register_activation
class LinQ(nn.Module):
    r"""
    Applies the LinQ (Linear Quadratic) activation function:

    :math:`\text{LinQ}(z) = \begin{cases} 
    az + 1 - 2z + z^2, & z \geq 2 - 2a \\ 
    \frac{1}{4}z(4 - |z|), & -2 + 2a < z < 2 - 2a \\ 
    az - 1 - 2z + z^2, & z \leq -2 + 2a 
    \end{cases}`

    Args:
        a (float, optional): Shape parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.LinQ(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.LinQ(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(LinQ, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        upper_threshold = 2 - 2 * self.a
        lower_threshold = -2 + 2 * self.a
        
        upper_region = z >= upper_threshold
        middle_region = (z > lower_threshold) & (z < upper_threshold)
        lower_region = z <= lower_threshold
        
        result[upper_region] = self.a * z[upper_region] + 1 - 2 * z[upper_region] + z[upper_region] ** 2
        result[middle_region] = 0.25 * z[middle_region] * (4 - torch.abs(z[middle_region]))
        result[lower_region] = self.a * z[lower_region] - 1 - 2 * z[lower_region] + z[lower_region] ** 2
        
        return result

    def _forward_inplace(self, z):
        upper_threshold = 2 - 2 * self.a
        lower_threshold = -2 + 2 * self.a
        
        upper_region = z >= upper_threshold
        middle_region = (z > lower_threshold) & (z < upper_threshold)
        lower_region = z <= lower_threshold
        
        z[upper_region] = self.a * z[upper_region] + 1 - 2 * z[upper_region] + z[upper_region] ** 2
        z[middle_region] = 0.25 * z[middle_region] * (4 - torch.abs(z[middle_region]))
        z[lower_region] = self.a * z[lower_region] - 1 - 2 * z[lower_region] + z[lower_region] ** 2
        
        return z


@register_activation
class ISRLU(nn.Module):
    r"""
    Applies the ISRLU (Inverse Square Root Linear Unit) activation function:

    :math:`\text{ISRLU}(z) = \begin{cases} 
    z, & z \geq 0 \\ 
    \frac{z}{\sqrt{1 + az^2}}, & z < 0 
    \end{cases}`

    Args:
        a (float, optional): Shape parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ISRLU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.ISRLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(ISRLU, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        pos = z >= 0
        neg = z < 0
        
        result[pos] = z[pos]
        result[neg] = z[neg] / torch.sqrt(1 + self.a * z[neg] ** 2)
        
        return result

    def _forward_inplace(self, z):
        neg = z < 0
        z[neg] = z[neg] / torch.sqrt(1 + self.a * z[neg] ** 2)
        return z

@register_activation
class ISRU(nn.Module):
    r"""
    Applies the ISRU (Inverse Square Root Unit) activation function:

    :math:`\text{ISRU}(z) = \frac{z}{\sqrt{1 + az^2}}`

    Args:
        a (float, optional): Shape parameter. Default: 1.0
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ISRU(a=0.5)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.ISRU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, a: float = 1.0, inplace: bool = False):
        super(ISRU, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return z / torch.sqrt(1 + self.a * z ** 2)

    def _forward_inplace(self, z):
        z.div_(torch.sqrt(1 + self.a * z ** 2))
        return z

@register_activation
class MEF(nn.Module):
    r"""
    Applies the MEF (Modified Error Function) activation function:

    :math:`\text{MEF}(z) = \frac{z}{\sqrt{1 + z^2} + 2}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.MEF()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.MEF(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(MEF, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        return z / (torch.sqrt(1 + z ** 2) + 2)

    def _forward_inplace(self, z):
        z.div_(torch.sqrt(1 + z ** 2) + 2)
        return z


@register_activation
class SquaredReLU(nn.Module):
    r"""
    Applies the SquaredReLU activation function:

    :math:`\text{SquaredReLU}(z) = \begin{cases} 
    z^2, & z > 0 \\ 
    0, & z \leq 0 
    \end{cases}`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SquaredReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SquaredReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    def forward(self, z) -> Tensor:
        return self._forward_inplace(z) if self.inplace else self._forward(z)

    def _forward(self, z):
        result = torch.empty_like(z)
        pos = z > 0
        neg = z <= 0
        
        result[pos] = z[pos] ** 2
        result[neg] = 0
        
        return result

    def _forward_inplace(self, z):
        pos = z > 0
        z[pos] = z[pos] ** 2
        return z