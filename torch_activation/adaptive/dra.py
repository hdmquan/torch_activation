import torch
import torch.nn as nn
from torch_activation.base import BaseActivation
from torch_activation import register_activation


@register_activation
class DRA(BaseActivation):
    r"""
    Dynamic Range Activator (DRA)

    y = x + (b · sin(a·x)²)/a + c · cos(a·x) + d · tanh(b·x)

    Learnable scalars **a, b, c, d** let the network adapt its dynamic
    range during training.

    Paper: *Can Transformers Do Enumerative Geometry?>*, ICLR 2025 – https://arxiv.org/abs/2408.14915
    """

    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))

    def _forward(self, x):
        term1 = x + (self.b * torch.square(torch.sin(self.a * x)) / self.a)
        term2 = self.c * torch.cos(self.a * x)
        term3 = self.d * torch.tanh(self.b * x)
        return term1 + term2 + term3

    def _forward_inplace(self, x):
        x.copy_(self._forward(x))
        return x


if __name__ == "__main__":
    from torch_activation.utils import plot_activation

    plot_activation(DRA, {"DRA": {}})
