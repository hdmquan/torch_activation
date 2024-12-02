# PyTorch Activations

PyTorch Activations is a collection of activation functions for the PyTorch library. This project aims to provide an easy-to-use solution for experimenting with different activation functions or simply adding variety to your models.

## Installation

You can install PyTorch Activations using pip:

```bash
$ pip install torch-activation
```

## Usage

To use the activation functions, import them from torch_activation. Here's an example:

```python
import torch_activation as tac

m = tac.ShiLU(inplace=True)
x = torch.rand(16, 3, 384, 384)
m(x)
```

Or in `nn.Sequential`:

```python
import torch
import torch.nn as nn
import torch_activation as tac

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 32, 2),
            tac.DELU(),
            nn.ConvTranspose2d(32, 64, 2),
            tac.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

Activation functions can be imported directly from the package, such as `torch_activation.CoLU`, or from submodules, such as `torch_activation.non_linear.CoLU`.

For a comprehensive list of available functions, please refer to the [LIST_OF_FUNCTION](LIST_OF_FUNCTION.md) file.

To learn more about usage, please refer to [Documentation](https://torch-activation.readthedocs.io)

We hope you find PyTorch Activations useful for your experimentation and model development. Enjoy exploring different activation functions!

## Contact

Alan Huynh - [LinkedIn](https://www.linkedin.com/in/hdmquan/) - [hdmquan@outlook.com](mailto:hdmquan@outlook.com)

Project Link: [https://github.com/hdmquan/torch_activation](https://github.com/hdmquan/torch_activation)

Documentation Link: [https://torch-activation.readthedocs.io](https://torch-activation.readthedocs.io)

PyPI Link: [https://pypi.org/project/torch-activation/](https://pypi.org/project/torch-activation/)
