import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class DELU(nn.Module):
    r"""
    Applies the DELU activation function:

    :math:`\text{DELU}(x) = \begin{cases} \text{SiLU}(x), x \leqslant 0 \\x(n-1), \text{otherwise} \end{cases}`


     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        n (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/DELU.png

    Examples:
        >>> m = nn.DELU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.DELU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, n: float = 1.0, inplace: bool = False):
        super(DELU, self).__init__()
        self.n = torch.nn.Parameter(Tensor([n]))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        return torch.where(
            x <= 0, F.silu(x), (self.n + 0.5) * x + torch.abs(torch.exp(-x) - 1)
        )

    def _forward_inplace(self, x):
        x[x <= 0] = F.silu(x[x <= 0])
        x[x > 0] = (self.n + 0.5) * x[x > 0] + torch.abs(torch.exp(-x[x > 0]) - 1)
        return x



class Phish(torch.nn.Module):
    r"""
    Applies the Phish activation function:

    :math:`\text{Phish}(x) = x \cdot \tanh (\text{GELU} (x))`

     See: `Phish: A Novel Hyper-Optimizable Activation Function`_.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/Phish.png

    Examples:
        >>> m = Phish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

    .. _`Phish: A Novel Hyper-Optimizable Activation Function`:
        https://www.semanticscholar.org/paper/Phish%3A-A-Novel-Hyper-Optimizable-Activation-Naveen/43eb5e22da6092d28f0e842fec53ec1a76e1ba6b
    """

    def __init__(self):
        super(Phish, self).__init__()

    def forward(self, x) -> Tensor:
        output = F.gelu(x)
        output = F.tanh(output)
        output = x * output
        return output


"""
class DReLUs(nn.Module):
    
    Applies the DReLUs activation function:

    :math:`\text{DReLUs}(x) = \begin{cases} \alpha (e ^ x - 1), x \leqslant 0 \\x, \text{otherwise} \end{cases}`

    Args:
        alpha (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples:
        >>> m = nn.DReLUs()
        >>> x = torch.randn(2)
        >>> output = m(x)
    
    
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)
        
    def _forward(self, x):
        return torch.where(x > 0, x,
                           self.alpha * (torch.exp(x) - 1))
        
    def _forward_inplace(self, x):
        x[x <= 0] = (torch.exp(x[x <= 0]) - 1) * self.alpha
        return x
 """
