import torch.nn as nn
from abc import ABC, abstractmethod

class BaseActivation(nn.Module, ABC):
    """
    Abstract base class for activation functions with optional in-place support.
    
    - If `inplace=True`, but the subclass does not implement `_forward_inplace()`, it raises `NotImplementedError`.
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    @abstractmethod
    def _forward(self, x):
        """Normal forward pass computation (required)."""
        pass  

    def _forward_inplace(self, x):
        """In-place computation (optional). If not implemented, it raises an error."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support in-place operations.")

    def forward(self, x):
        """Main dispatch method."""
        return self._forward_inplace(x) if self.inplace else self._forward(x)


# Example usage
"""
@register_activation
class MyActivation(BaseActivation):
    '''
    My activation function. Define as:
    
    :math:`y = x * 2`

    Args:
        inplace (bool, optional): If True, the operation is done in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/MyActivation.png

    Examples::
        >>> m = MyActivation()
        >>> x = torch.randn(2)
        >>> output = m(x)
    '''
    def __init__(self, inplace=False, **kwargs):
        super().__init__(inplace, **kwargs)

    def _forward(self, x):
        return x * 2

    def _forward_inplace(self, x):
        x *= 2
"""
