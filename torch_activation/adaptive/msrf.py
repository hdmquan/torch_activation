import torch
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class MollifiedAbsoluteValue(BaseActivation):
    r"""
    Applies the Mollified Absolute Value activation function:

    :math:`\text{MollifiedAbsoluteValue}(x) = \sqrt{x^2 + \epsilon}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MollifiedAbsoluteValue.png

    Examples::

        >>> m = torch_activation.MollifiedAbsoluteValue()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MollifiedAbsoluteValue(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        return torch.sqrt(x.pow(2) + self.epsilon)


@register_activation
class SquarePlus(BaseActivation):
    r"""
    Applies the SquarePlus activation function:

    :math:`\text{SquarePlus}(z) = \frac{1}{2} (z + \sqrt{z^2 + \epsilon})`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SquarePlus.png

    Examples::

        >>> m = torch_activation.SquarePlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SquarePlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        return 0.5 * (x + torch.sqrt(x.pow(2) + self.epsilon))


@register_activation
class StepPlus(BaseActivation):
    r"""
    Applies the StepPlus activation function:

    :math:`\text{StepPlus}(z) = \frac{1}{2} \left(1 + \frac{z}{\sqrt{z^2 + \epsilon}}\right)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/StepPlus.png

    Examples::

        >>> m = torch_activation.StepPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.StepPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return 0.5 * (1 + x / abs_x_eps)


@register_activation
class BipolarPlus(BaseActivation):
    r"""
    Applies the BipolarPlus activation function:

    :math:`\text{BipolarPlus}(z) = \frac{z}{\sqrt{z^2 + \epsilon}}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/BipolarPlus.png

    Examples::

        >>> m = torch_activation.BipolarPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.BipolarPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return x / abs_x_eps


@register_activation
class LReLUPlus(BaseActivation):
    r"""
    Applies the Leaky ReLU Plus activation function:

    :math:`\text{LReLUPlus}(z_i) = \frac{1}{2} (z_i + a_i z_i + \sqrt{((1 - a_i) z_i)^2 + \epsilon})`

    Args:
        negative_slope (float, optional): Controls the angle of the negative slope. Default: ``0.01``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LReLUPlus.png

    Examples::

        >>> m = torch_activation.LReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LReLUPlus(negative_slope=0.1)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, negative_slope=0.01, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.negative_slope = negative_slope
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        term = (1 - self.negative_slope) * x
        abs_term_eps = torch.sqrt(term.pow(2) + self.epsilon)
        return 0.5 * (x + self.negative_slope * x + abs_term_eps)


@register_activation
class vReLUPlus(BaseActivation):
    r"""
    Applies the vReLU Plus activation function:

    :math:`\text{vReLUPlus}(z) = \sqrt{z^2 + \epsilon}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/vReLUPlus.png

    Examples::

        >>> m = torch_activation.vReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.vReLUPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        return torch.sqrt(x.pow(2) + self.epsilon)


@register_activation
class SoftshrinkPlus(BaseActivation):
    r"""
    Applies the Softshrink Plus activation function:

    :math:`\text{SoftshrinkPlus}(z) = z + \frac{1}{2} \left(\sqrt{(z - a)^2 + \epsilon} - \sqrt{(z + a)^2 + \epsilon}\right)`

    Args:
        lambda_val (float, optional): The lambda value for the Softshrink formulation. Default: ``0.5``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SoftshrinkPlus.png

    Examples::

        >>> m = torch_activation.SoftshrinkPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SoftshrinkPlus(lambda_val=1.0)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, lambda_val=0.5, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.lambda_val = lambda_val
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        term1 = torch.sqrt((x - self.lambda_val).pow(2) + self.epsilon)
        term2 = torch.sqrt((x + self.lambda_val).pow(2) + self.epsilon)
        return x + 0.5 * (term1 - term2)


@register_activation
class PanPlus(BaseActivation):
    r"""
    Applies the Pan Plus activation function:

    :math:`\text{PanPlus}(z) = -a + \frac{1}{2} \left(\sqrt{(z - a)^2 + \epsilon} + \sqrt{(z + a)^2 + \epsilon}\right)`

    Args:
        a (float, optional): The 'a' parameter in the Pan Plus formulation. Default: ``0.5``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/PanPlus.png

    Examples::

        >>> m = torch_activation.PanPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.PanPlus(a=1.0)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a=0.5, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        term1 = torch.sqrt((x - self.a).pow(2) + self.epsilon)
        term2 = torch.sqrt((x + self.a).pow(2) + self.epsilon)
        return -self.a + 0.5 * (term1 + term2)


@register_activation
class BReLUPlus(BaseActivation):
    r"""
    Applies the Bounded ReLU Plus activation function:

    :math:`\text{BReLUPlus}(z) = \frac{1}{2} (1 + \sqrt{z^2 + \epsilon} - \sqrt{(z - 1)^2 + \epsilon})`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/BReLUPlus.png

    Examples::

        >>> m = torch_activation.BReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.BReLUPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        abs_x_minus_1_eps = torch.sqrt((x - 1).pow(2) + self.epsilon)
        return 0.5 * (1 + abs_x_eps - abs_x_minus_1_eps)


@register_activation
class SReLUPlus(BaseActivation):
    r"""
    Applies the S-shaped ReLU Plus activation function:

    :math:`\text{SReLUPlus}(z_i) = a_i z_i + \frac{1}{2} (a_i - 1) (\sqrt{(z_i - t_i)^2 + \epsilon} - \sqrt{(z_i + t_i)^2 + \epsilon})`

    Args:
        a (float, optional): The 'a' parameter in the SReLU Plus formulation. Default: ``0.5``
        t (float, optional): The 't' parameter in the SReLU Plus formulation. Default: ``1.0``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SReLUPlus.png

    Examples::

        >>> m = torch_activation.SReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SReLUPlus(a=0.2, t=2.0)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a=0.5, t=1.0, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.t = t
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_minus_t_eps = torch.sqrt((x - self.t).pow(2) + self.epsilon)
        abs_x_plus_t_eps = torch.sqrt((x + self.t).pow(2) + self.epsilon)
        return self.a * x + 0.5 * (self.a - 1) * (abs_x_minus_t_eps - abs_x_plus_t_eps)


@register_activation
class HardTanhPlus(BaseActivation):
    r"""
    Applies the HardTanh Plus activation function:

    :math:`\text{HardTanhPlus}(z) = \frac{1}{2} (\sqrt{(z + 1)^2 + \epsilon} - \sqrt{(z - 1)^2 + \epsilon})`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardTanhPlus.png

    Examples::

        >>> m = torch_activation.HardTanhPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HardTanhPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_plus_1_eps = torch.sqrt((x + 1).pow(2) + self.epsilon)
        abs_x_minus_1_eps = torch.sqrt((x - 1).pow(2) + self.epsilon)
        return 0.5 * (abs_x_plus_1_eps - abs_x_minus_1_eps)


@register_activation
class HardshrinkPlus(BaseActivation):
    r"""
    Applies the Hardshrink Plus activation function:

    :math:`\text{HardshrinkPlus}(z) = z \left(1 + \frac{1}{2} \left(\frac{z - a}{\sqrt{(z - a)^2 + \epsilon}} - \frac{z + a}{\sqrt{(z + a)^2 + \epsilon}}\right)\right)`

    Args:
        lambda_val (float, optional): The lambda value for the Hardshrink formulation. Default: ``0.5``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/HardshrinkPlus.png

    Examples::

        >>> m = torch_activation.HardshrinkPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.HardshrinkPlus(lambda_val=1.0)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, lambda_val=0.5, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.lambda_val = lambda_val
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        term1 = (x - self.lambda_val) / torch.sqrt((x - self.lambda_val).pow(2) + self.epsilon)
        term2 = (x + self.lambda_val) / torch.sqrt((x + self.lambda_val).pow(2) + self.epsilon)
        return x * (1 + 0.5 * (term1 - term2))


@register_activation
class MollifiedMeLUComponent(BaseActivation):
    r"""
    Applies the Mollified MeLU Component activation function:

    :math:`\text{MollifiedMeLUComponent}(z_i) = \frac{1}{2} \left(c - \sqrt{(z_i - b)^2 + \epsilon} + \sqrt{\left(c - \sqrt{(z_i - b)^2 + \epsilon}\right)^2 + \epsilon}\right)`

    Args:
        b (float, optional): The 'b' parameter in the MeLU formulation. Default: ``0.0``
        c (float, optional): The 'c' parameter in the MeLU formulation. Default: ``1.0``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MollifiedMeLUComponent.png

    Examples::

        >>> m = torch_activation.MollifiedMeLUComponent()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MollifiedMeLUComponent(b=0.5, c=2.0)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, b=0.0, c=1.0, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.b = b
        self.c = c
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_minus_b_eps = torch.sqrt((x - self.b).pow(2) + self.epsilon)
        term = self.c - abs_x_minus_b_eps
        return 0.5 * (term + torch.sqrt(term.pow(2) + self.epsilon))


@register_activation
class TSAFPlus(BaseActivation):
    r"""
    Applies the TSAF Plus activation function:

    :math:`\text{TSAFPlus}(z_i) = \frac{1}{4} \left(\sqrt{(z_i - a + c)^2 + \epsilon} + \sqrt{(z_i - a)^2 + \epsilon} + \sqrt{(z_i + b - c)^2 + \epsilon} - \sqrt{(z_i - b)^2 + \epsilon}\right)`

    Args:
        a (float, optional): The 'a' parameter in the TSAF formulation. Default: ``0.5``
        b (float, optional): The 'b' parameter in the TSAF formulation. Default: ``0.5``
        c (float, optional): The 'c' parameter in the TSAF formulation. Default: ``1.0``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/TSAFPlus.png

    Examples::

        >>> m = torch_activation.TSAFPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.TSAFPlus(a=1.0, b=0.5, c=2.0)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, a=0.5, b=0.5, c=1.0, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        term1 = torch.sqrt((x - self.a + self.c).pow(2) + self.epsilon)
        term2 = torch.sqrt((x - self.a).pow(2) + self.epsilon)
        term3 = torch.sqrt((x + self.b - self.c).pow(2) + self.epsilon)
        term4 = torch.sqrt((x - self.b).pow(2) + self.epsilon)
        return 0.25 * (term1 + term2 + term3 - term4)


@register_activation
class ELUPlus(BaseActivation):
    r"""
    Applies the ELU Plus activation function:

    :math:`\text{ELUPlus}(z) = \frac{1}{2} (z + \sqrt{z^2 + \epsilon}) + \frac{1}{2} \left(\frac{\exp(z) - 1}{\alpha} + \sqrt{\left(\frac{\exp(z) - 1}{\alpha}\right)^2 + \epsilon}\right)`

    Args:
        alpha (float, optional): The alpha value for the ELU formulation. Default: ``1.0``
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ELUPlus.png

    Examples::

        >>> m = torch_activation.ELUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ELUPlus(alpha=0.5)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, alpha=1.0, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        term1 = 0.5 * (x + torch.sqrt(x.pow(2) + self.epsilon))
        elu_term = (torch.exp(x.clamp(max=44.0)) - 1) / self.alpha
        term2 = 0.5 * (elu_term + torch.sqrt(elu_term.pow(2) + self.epsilon))
        return term1 + term2


@register_activation
class SwishPlus(BaseActivation):
    r"""
    Applies the Swish Plus activation function:

    :math:`\text{SwishPlus}(z) = \frac{1}{2} \left(z + \frac{z^2}{\sqrt{z^2 + \epsilon}}\right)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SwishPlus.png

    Examples::

        >>> m = torch_activation.SwishPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SwishPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return 0.5 * (x + (x.pow(2) / abs_x_eps))


@register_activation
class MishPlus(BaseActivation):
    r"""
    Applies the Mish Plus activation function:

    :math:`\text{MishPlus}(z) = z \cdot \text{BipolarPlus}(\text{BipolarPlus}(z))`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MishPlus.png

    Examples::

        >>> m = torch_activation.MishPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.MishPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        # First BipolarPlus
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        bipolar1 = x / abs_x_eps

        # Second BipolarPlus
        abs_bipolar1_eps = torch.sqrt(bipolar1.pow(2) + self.epsilon)
        bipolar2 = bipolar1 / abs_bipolar1_eps

        return x * bipolar2


@register_activation
class LogishPlus(BaseActivation):
    r"""
    Applies the Logish Plus activation function:

    :math:`\text{LogishPlus}(z) = z \cdot \ln\left(1 + \frac{1}{2}\left(1 + \frac{z}{\sqrt{z^2 + \epsilon}}\right)\right)`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/LogishPlus.png

    Examples::

        >>> m = torch_activation.LogishPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.LogishPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        step_plus = 0.5 * (1 + x / abs_x_eps)
        return x * torch.log1p(step_plus)


@register_activation
class SoftsignPlus(BaseActivation):
    r"""
    Applies the Softsign Plus activation function:

    :math:`\text{SoftsignPlus}(z) = \frac{z}{1 + \sqrt{z^2 + \epsilon}}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SoftsignPlus.png

    Examples::

        >>> m = torch_activation.SoftsignPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SoftsignPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        return x / (1 + abs_x_eps)


@register_activation
class SignReLUPlus(BaseActivation):
    r"""
    Applies the SignReLU Plus activation function:

    :math:`\text{SignReLUPlus}(z) = \frac{1}{2} (z + \sqrt{z^2 + \epsilon}) + \frac{z - \sqrt{z^2 + \epsilon}}{2\sqrt{(1 - z)^2 + \epsilon}}`

    Args:
        epsilon (float, optional): Small constant for numerical stability. Default: ``1e-6``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SignReLUPlus.png

    Examples::

        >>> m = torch_activation.SignReLUPlus()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SignReLUPlus(epsilon=1e-4)
        >>> x = torch.randn(2, 3, 4)
        >>> output = m(x)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _forward(self, x) -> Tensor:
        abs_x_eps = torch.sqrt(x.pow(2) + self.epsilon)
        abs_1_minus_x_eps = torch.sqrt((1 - x).pow(2) + self.epsilon)
        term1 = 0.5 * (x + abs_x_eps)
        term2 = (x - abs_x_eps) / (2 * abs_1_minus_x_eps)
        return term1 + term2
