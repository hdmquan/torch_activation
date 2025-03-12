# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_activation.base import BaseActivation

# from torch import Tensor
# from typing import Iterable


# class LinComb(BaseActivation):
#     r"""
#     Applies the LinComb activation function:

#     :math:`\text{LinComb}(x) = \sum_{i=1}^{n} w_i \cdot F_i(x)`

#      See: https://doi.org/10.20944/preprints202301.0463.v1

#     Args:
#         activations (Iterable[nn.Module]): List of activation functions. Default: [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softsign]

#     Shape:
#         - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
#         - Output: :math:`(*)`

#     Here is a plot of the function and its derivative:

#     .. image:: ../images/activation_images/LinComb.png

#     Examples::

#         >>> activations = [nn.ReLU(), nn.Sigmoid()]
#         >>> m = LinComb(activation_functions)
#         >>> input = torch.randn(10)
#         >>> output = m(input)
#     """

#     def __init__(
#         self,
#         activations: Iterable[nn.Module] = [
#             nn.ReLU(),
#             nn.Sigmoid(),
#             nn.Tanh(),
#             nn.Softsign(),
#         ],
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.activations = nn.ModuleList(activations)
#         self.weights = nn.Parameter(Tensor(len(activations)))

#         self.weights.data.uniform_(-1, 1)

#     def _forward(self, input) -> Tensor:
#         activations = [
#             self.weights[i] * self.activations[i](input)
#             for i in range(len(self.activations))
#         ]
#         return torch.sum(torch.stack(activations), dim=0)


# class NormLinComb(BaseActivation):
#     r"""
#     Applies the LinComb activation function:

#     :math:`\text{NormLinComb}(x) = \frac{\sum_{i=1}^{n} w_i \cdot F_i(x)}{\|\|W\|\|}`

#      See: https://doi.org/10.20944/preprints202301.0463.v1

#     Args:
#         activations (Iterable[nn.Module]): List of activation functions. Default: [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softsign]

#     Shape:
#         - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
#         - Output: :math:`(*)`

#     Here is a plot of the function and its derivative:

#     .. image:: ../images/activation_images/NormLinComb.png

#     Examples::

#         >>> activations = [nn.ReLU, nn.Sigmoid]
#         >>> m = NormLinComb(activation_functions)
#         >>> input = torch.randn(10)
#         >>> output = m(input)
#     """

#     def __init__(
#         self,
#         activations: Iterable[nn.Module] = [
#             nn.ReLU(),
#             nn.Sigmoid(),
#             nn.Tanh(),
#             nn.Softsign(),
#         ],
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.activations = nn.ModuleList(activations)
#         self.weights = nn.Parameter(torch.Tensor(len(activations)))

#         self.weights.data.uniform_(-1, 1)

#     def _forward(self, input) -> torch.Tensor:
#         activations = [
#             self.weights[i] * self.activations[i](input)
#             for i in range(len(self.activations))
#         ]
#         output = torch.sum(torch.stack(activations), dim=0)
#         return output / torch.norm(output)

