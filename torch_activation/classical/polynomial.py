from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class NCU(BaseActivation):
    r"""
    Applies the Non-monotonic Cubic Unit (NCU) activation function:

    :math:`\text{NCU}(z) = z - z^3`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/NCU.png

    Examples::

        >>> m = torch_activation.NCU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Unused

    def _forward(self, z) -> Tensor:
        return z - z**3


@register_activation
class Triple(BaseActivation):
    r"""
    Applies the Triple activation function:

    :math:`\text{Triple}(z) = a \cdot z^3`

    Args:
        a (float, optional): parameter for the cubic term. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Triple.png

    Examples::

        >>> m = torch_activation.Triple()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        # Unused

    def _forward(self, z) -> Tensor:
        return self.a * z**3


@register_activation
class SQU(BaseActivation):
    r"""
    Applies the Shifted Quadratic Unit (SQU) activation function:

    :math:`\text{SQU}(z) = z^2 + z`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/SQU.png

    Examples::

        >>> m = torch_activation.SQU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Unused

    def _forward(self, z) -> Tensor:
        return z**2 + z
