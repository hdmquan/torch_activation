Examples
========

.. code-block:: python
    :caption: Example

    import torch
    import torch_activation as tac

    x = torch.rand(3, 20, 20)
    m = tac.SinLU()

    y_ = m(x)


.. code-block:: python
    :caption: Example when using in nn.Sequential

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
