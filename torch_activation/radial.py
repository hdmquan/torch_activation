import torch

# import torch.nn as nn
# import torch.nn.functional as F

from torch import Tensor


class ScaledSoftSign(torch.nn.Module):
    r"""
    Applies the ScaledSoftSign activation function:

    :math:`\text{ScaledSoftSign}(x) = \frac{\alpha \cdot x}{\beta + \|x\|}`

     See: https://doi.org/10.20944/preprints202301.0463.v1

    Args:
        alpha (float, optional): The initial value of the alpha parameter. Default: 1.0
        beta (float, optional): The initial value of the beta parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Here is a plot of the function and its derivative:

    .. image:: ../images/activation_images/ScaledSoftSign.png

    Examples:
        >>> m = ScaledSoftSign(alpha=0.5, beta=1.0)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = ScaledSoftSign(inplace=True)
        >>> x = torch.randn(2, 3)
        >>> m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):

        super(ScaledSoftSign, self).__init__()

        self.alpha = torch.nn.Parameter(Tensor([alpha]))
        self.beta = torch.nn.Parameter(Tensor([beta]))

    def forward(self, x) -> Tensor:
        abs_x = x.abs()
        alpha_x = self.alpha * x
        denom = self.beta + abs_x
        result = alpha_x / denom
        return result


if __name__ == "__main__":
    pass
