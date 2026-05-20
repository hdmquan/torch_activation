.. _activations-ActivationName:

ActivationName
==============

One-line description.

.. math::

   \text{ActivationName}(x) = \text{formula}

.. autoclass:: torch_activation.ActivationName
   :members:
   :undoc-members:
   :show-inheritance:

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``inplace``
     - ``False``
     - Performs the operation in-place if supported.
   * - ``param_name``
     - ``default``
     - Parameter description.

Properties
----------

.. list-table::
   :widths: 30 70

   * - Differentiable
     - Yes / No
   * - Monotonic
     - Yes / No
   * - Bounded
     - Yes (range: [a, b]) / No
   * - Output range
     - :math:`(-\infty, +\infty)`

Example
-------

.. code-block:: python

   import torch
   import torch_activation

   m = torch_activation.ActivationName()
   x = torch.randn(2, 3)
   output = m(x)

References
----------

.. [CITE_KEY] Author(s). *Paper Title*. Venue, Year.
   `arXiv:XXXX.XXXXX <https://arxiv.org/abs/XXXX.XXXXX>`_
