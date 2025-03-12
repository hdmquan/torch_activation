# Contributing to torch_activation

Thank you for your interest in contributing to torch_activation! We welcome all contributions to help make this project better.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Install dependencies:
```bash
poetry install
```

## Revise an Existing Activation Function

To revise an existing activation function:

1. Try to only change the functions in question
2. Make your changes while ensuring:
   - Documentation remains accurate and complete
   - Mathematical formulas are updated if needed
   - References and citations are maintained/updated
   - The function's behavior is clearly described
   - Add/Update activation plot if needed
3. Update tests if the changes affect the function's behavior
4. Run the test files.
5. If you've modified the function's behavior significantly:
   - Update the function's docstring
   - Update or add new test cases
   - Regenerate the function plot if applicable

## Adding a New Activation Function

### 1. Choose the Right Category

We have two main categories of activation functions:

- **Classical**: Traditional activation functions that apply a fixed transformation (e.g., ReLU, Sigmoid, Tanh)
- **Adaptive**: Activation functions with learnable parameters or dynamic behavior

Try to find a file where similar functions reside. If you're unsure which category best fits your activation function, please:
1. Open an issue on GitHub
2. Describe your activation function and its behavior
3. We'll discuss the best placement together!

### 2. Implementation Steps

1. Create your activation class in the appropriate folder (`adaptive/` or `classical/`)
2. Inherit from `BaseActivation` class
3. Implement the required methods:
   - `_forward()` (required)
   - `_forward_inplace()` (optional, for in-place operations if can_be_inplace function in utils return `True`)

Example:
```python
from torch_activation.base import BaseActivation
@register_activation
class MyActivation(BaseActivation):
    '''
    My activation function. Define as:
    
    :math:`y = x * 2`

    See: `https://arxiv.org/...`

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
```

### 3. Testing

Before submitting a PR:
1. Run all tests in the `tests/` folder:
```bash
pytest tests/
```
2. Ensure all tests pass
3. Add tests for your activation function in the appropriate test file

## Creating a Pull Request

1. Commit your changes
2. Push to your fork
3. Create a Pull Request to the main repository
4. In the PR description, include:
   - Brief description of the activation function
   - Mathematical formula (if applicable)
   - Any relevant references or papers

We'll review your PR and provide feedback as needed. Thank you for contributing!
