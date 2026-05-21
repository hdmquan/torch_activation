import torch
from torch import Tensor

from torch_activation import register_activation
from torch_activation.base import BaseActivation


@register_activation
class ETanh(BaseActivation):
    r"""
    Applies the E-Tanh activation function:

    :math:`\text{ETanh}(z) = a \cdot \exp(z) \cdot \tanh(z)`

    Args:
        a (float, optional): Scaling parameter. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ETanh.png

    Examples::

        >>> m = torch_activation.ETanh()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ETanh(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, z) -> Tensor:
        return self.a * torch.exp(z.clamp(max=88.0)) * torch.tanh(z)


@register_activation
class EvolvedTanhReLU(BaseActivation):
    r"""
    Applies the evolved combination of tanh and ReLU activation function:

    :math:`\text{EvolvedTanhReLU}(z) = a \cdot \tanh(z^2) + \text{ReLU}(z)`

    Args:
        a (float, optional): Scaling parameter. Default: ``1.0``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/EvolvedTanhReLU.png

    Examples::

        >>> m = torch_activation.EvolvedTanhReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.EvolvedTanhReLU(a=2.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, a: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def _forward(self, z) -> Tensor:
        return self.a * torch.tanh(z**2) + torch.relu(z)


@register_activation
class EvolvedTanhLogReLU(BaseActivation):
    r"""
    Applies the evolved regular activation function combining tanh and ReLU:

    :math:`\text{EvolvedTanhLogReLU}(z) = \max(\tanh(\log(z)), \text{ReLU}(z))`

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/EvolvedTanhLogReLU.png

    Examples::

        >>> m = torch_activation.EvolvedTanhLogReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, z) -> Tensor:
        # Handle potential negative values for log
        safe_log = torch.log(torch.clamp(z, min=1e-10))
        return torch.maximum(torch.tanh(safe_log), torch.relu(z))
