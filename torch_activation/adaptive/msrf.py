import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_activation import register_activation

@register_activation
class MollifiedAbsoluteValue(nn.Module):
    r"""
    Applies the Mollified Absolute Value function:

    :math:`|x|_\epsilon = \sqrt{x^2 + \epsilon}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MollifiedAbsoluteValue()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(MollifiedAbsoluteValue, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        return torch.sqrt(x.pow(2) + self.epsilon)


@register_activation
class SquarePlus(nn.Module):
    r"""
    Applies the SquarePlus function:

    :math:`\text{SquarePlus}(z) = \frac{1}{2} (z + |z|_\epsilon) = \frac{1}{2} (z + \sqrt{z^2 + \epsilon})`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SquarePlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(SquarePlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        return 0.5 * (x + torch.sqrt(x.pow(2) + self.epsilon))


@register_activation
class StepPlus(nn.Module):
    r"""
    Applies the StepPlus function:

    :math:`\text{StepPlus}(z) = \frac{1}{2} \left(1 + \frac{z}{|z|_\epsilon}\right)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = StepPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(StepPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return 0.5 * (1 + x / abs_x_eps)


@register_activation
class BipolarPlus(nn.Module):
    r"""
    Applies the BipolarPlus function:

    :math:`\text{BipolarPlus}(z) = \frac{z}{|z|_\epsilon}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = BipolarPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(BipolarPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return x / abs_x_eps


@register_activation
class LReLUPlus(nn.Module):
    r"""
    Applies the Leaky ReLU Plus function:

    :math:`\text{LReLUPlus}(z_i) = \frac{1}{2} (z_i + a_i z_i + |(1 - a_i) z_i|_\epsilon)`

    Args:
        negative_slope (float or Tensor, optional): Controls the angle of the negative slope. Default: 0.01
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, negative_slope=0.01, epsilon=1e-6):
        super(LReLUPlus, self).__init__()
        self.negative_slope = negative_slope
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        term = (1 - self.negative_slope) * x
        abs_term_eps = torch.sqrt(term.pow(2) + self.epsilon)
        return 0.5 * (x + self.negative_slope * x + abs_term_eps)


@register_activation
class vReLUPlus(nn.Module):
    r"""
    Applies the vReLU Plus function:

    :math:`\text{vReLUPlus}(z) = |z|_\epsilon`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = vReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(vReLUPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        return torch.sqrt(x.pow(2) + self.epsilon)


@register_activation
class SoftshrinkPlus(nn.Module):
    r"""
    Applies the Softshrink Plus function:

    :math:`\text{SoftshrinkPlus}(z) = z + \frac{1}{2} \left(\sqrt{(z - a)^2 + \epsilon} - \sqrt{(z + a)^2 + \epsilon}\right)`

    Args:
        lambda_val (float, optional): The lambda value for the Softshrink formulation. Default: 0.5
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SoftshrinkPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, lambda_val=0.5, epsilon=1e-6):
        super(SoftshrinkPlus, self).__init__()
        self.lambda_val = lambda_val
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        term1 = torch.sqrt((x - self.lambda_val).pow(2) + self.epsilon)
        term2 = torch.sqrt((x + self.lambda_val).pow(2) + self.epsilon)
        return x + 0.5 * (term1 - term2)


@register_activation
class PanPlus(nn.Module):
    r"""
    Applies the Pan Plus function:

    :math:`\text{PanPlus}(z) = -a + \frac{1}{2} \left(\sqrt{(z - a)^2 + \epsilon} + \sqrt{(z + a)^2 + \epsilon}\right)`

    Args:
        a (float, optional): The 'a' parameter in the Pan Plus formulation. Default: 0.5
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = PanPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a=0.5, epsilon=1e-6):
        super(PanPlus, self).__init__()
        self.a = a
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        term1 = torch.sqrt((x - self.a).pow(2) + self.epsilon)
        term2 = torch.sqrt((x + self.a).pow(2) + self.epsilon)
        return -self.a + 0.5 * (term1 + term2)


@register_activation
class BReLUPlus(nn.Module):
    r"""
    Applies the Bounded ReLU Plus function:

    :math:`\text{BReLUPlus}(z) = \frac{1}{2} (1 + |z|_\epsilon - |z - 1|_\epsilon)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = BReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(BReLUPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        abs_x_minus_1_eps = torch.sqrt((x - 1).pow(2) + self.epsilon)
        return 0.5 * (1 + abs_x_eps - abs_x_minus_1_eps)


@register_activation
class SReLUPlus(nn.Module):
    r"""
    Applies the S-shaped ReLU Plus function:

    :math:`\text{SReLUPlus}(z_i) = a_i z_i + \frac{1}{2} (a_i - 1) (|z_i - t_i|_\epsilon - |z_i + t_i|_\epsilon)`

    Args:
        a (float or Tensor, optional): The 'a' parameter in the SReLU Plus formulation. Default: 0.5
        t (float or Tensor, optional): The 't' parameter in the SReLU Plus formulation. Default: 1.0
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a=0.5, t=1.0, epsilon=1e-6):
        super(SReLUPlus, self).__init__()
        self.a = a
        self.t = t
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_minus_t_eps = torch.sqrt((x - self.t).pow(2) + self.epsilon)
        abs_x_plus_t_eps = torch.sqrt((x + self.t).pow(2) + self.epsilon)
        return self.a * x + 0.5 * (self.a - 1) * (abs_x_minus_t_eps - abs_x_plus_t_eps)


@register_activation
class HardTanhPlus(nn.Module):
    r"""
    Applies the HardTanh Plus function:

    :math:`\text{HardTanhPlus}(z) = \frac{1}{2} (|z + 1|_\epsilon - |z - 1|_\epsilon)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardTanhPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(HardTanhPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_plus_1_eps = torch.sqrt((x + 1).pow(2) + self.epsilon)
        abs_x_minus_1_eps = torch.sqrt((x - 1).pow(2) + self.epsilon)
        return 0.5 * (abs_x_plus_1_eps - abs_x_minus_1_eps)


@register_activation
class HardshrinkPlus(nn.Module):
    r"""
    Applies the Hardshrink Plus function:

    :math:`\text{HardshrinkPlus}(z) = z \left(1 + \frac{1}{2} \left(\frac{z - a}{\sqrt{(z - a)^2 + \epsilon}} - \frac{z + a}{\sqrt{(z + a)^2 + \epsilon}}\right)\right)`

    Args:
        lambda_val (float, optional): The lambda value for the Hardshrink formulation. Default: 0.5
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardshrinkPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, lambda_val=0.5, epsilon=1e-6):
        super(HardshrinkPlus, self).__init__()
        self.lambda_val = lambda_val
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        term1 = (x - self.lambda_val) / torch.sqrt((x - self.lambda_val).pow(2) + self.epsilon)
        term2 = (x + self.lambda_val) / torch.sqrt((x + self.lambda_val).pow(2) + self.epsilon)
        return x * (1 + 0.5 * (term1 - term2))


@register_activation
class MollifiedMeLUComponent(nn.Module):
    r"""
    Applies the Mollified MeLU Component function:

    :math:`\phi_{b_j c_j \text{Plus}}(z_i) = \frac{1}{2} \left(c_j - |z_i - b_j|_\epsilon + \sqrt{(c_j - |z_i - b_j|_\epsilon)^2 + \epsilon}\right)`

    Args:
        b (float, optional): The 'b' parameter in the MeLU formulation. Default: 0.0
        c (float, optional): The 'c' parameter in the MeLU formulation. Default: 1.0
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MollifiedMeLUComponent()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, b=0.0, c=1.0, epsilon=1e-6):
        super(MollifiedMeLUComponent, self).__init__()
        self.b = b
        self.c = c
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_minus_b_eps = torch.sqrt((x - self.b).pow(2) + self.epsilon)
        term = self.c - abs_x_minus_b_eps
        return 0.5 * (term + torch.sqrt(term.pow(2) + self.epsilon))


@register_activation
class TSAFPlus(nn.Module):
    r"""
    Applies the TSAF Plus function:

    :math:`\text{TSAFPlus}(z_i) = \frac{1}{4} \left(|z_i - a_i + c_i|_\epsilon + |z_i - a_i|_\epsilon + |z_i + b_i - c_i|_\epsilon - |z_i - b_i|_\epsilon\right)`

    Args:
        a (float, optional): The 'a' parameter in the TSAF formulation. Default: 0.5
        b (float, optional): The 'b' parameter in the TSAF formulation. Default: 0.5
        c (float, optional): The 'c' parameter in the TSAF formulation. Default: 1.0
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = TSAFPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a=0.5, b=0.5, c=1.0, epsilon=1e-6):
        super(TSAFPlus, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        term1 = torch.sqrt((x - self.a + self.c).pow(2) + self.epsilon)
        term2 = torch.sqrt((x - self.a).pow(2) + self.epsilon)
        term3 = torch.sqrt((x + self.b - self.c).pow(2) + self.epsilon)
        term4 = torch.sqrt((x - self.b).pow(2) + self.epsilon)
        return 0.25 * (term1 + term2 + term3 - term4)


@register_activation
class ELUPlus(nn.Module):
    r"""
    Applies the ELU Plus function:

    :math:`\text{ELUPlus}(z) = \frac{1}{2} (z + |z|_\epsilon) + \frac{1}{2} \left(\frac{\exp(z) - 1}{a} + \sqrt{\left(\frac{\exp(z) - 1}{a}\right)^2 + \epsilon}\right)`

    Args:
        alpha (float, optional): The alpha value for the ELU formulation. Default: 1.0
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ELUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, alpha=1.0, epsilon=1e-6):
        super(ELUPlus, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        term1 = 0.5 * (x + torch.sqrt(x.pow(2) + self.epsilon))
        elu_term = (torch.exp(x) - 1) / self.alpha
        term2 = 0.5 * (elu_term + torch.sqrt(elu_term.pow(2) + self.epsilon))
        return term1 + term2


@register_activation
class SwishPlus(nn.Module):
    r"""
    Applies the Swish Plus function:

    :math:`\text{SwishPlus}(z) = z \cdot \text{StepPlus}(z) = \frac{1}{2} \left(z + \frac{z^2}{|z|_\epsilon}\right)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SwishPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(SwishPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return 0.5 * (x + (x.pow(2) / abs_x_eps))


@register_activation
class MishPlus(nn.Module):
    r"""
    Applies the Mish Plus function:

    :math:`\text{MishPlus}(z) = z \cdot \text{BipolarPlus}(\text{BipolarPlus}(z))`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = MishPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(MishPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        # First BipolarPlus
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        bipolar1 = x / abs_x_eps
        
        # Second BipolarPlus
        abs_bipolar1_eps = torch.sqrt(bipolar1.pow(2) + self.epsilon)
        bipolar2 = bipolar1 / abs_bipolar1_eps
        
        return x * bipolar2


@register_activation
class LogishPlus(nn.Module):
    r"""
    Applies the Logish Plus function:

    :math:`\text{LogishPlus}(z) = z \cdot \ln(1 + \text{StepPlus}(z))`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = LogishPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(LogishPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        step_plus = 0.5 * (1 + x / abs_x_eps)
        return x * torch.log(1 + step_plus)


@register_activation
class SoftsignPlus(nn.Module):
    r"""
    Applies the Softsign Plus function:

    :math:`\text{SoftsignPlus}(z) = \frac{z}{1 + |z|_\epsilon}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SoftsignPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(SoftsignPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return x / (1 + abs_x_eps)


@register_activation
class SignReLUPlus(nn.Module):
    r"""
    Applies the SignReLU Plus function:

    :math:`\text{SignReLUPlus}(z) = \frac{1}{2} (z + |z|_\epsilon) + \frac{z - |z|_\epsilon}{2 |1 - z|_\epsilon}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SignReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6):
        super(SignReLUPlus, self).__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        abs_1_minus_x_eps = torch.sqrt((1 - x).pow(2) + self.epsilon)
        term1 = 0.5 * (x + abs_x_eps)
        term2 = (x - abs_x_eps) / (2 * abs_1_minus_x_eps)
        return term1 + term2
