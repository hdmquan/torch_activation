# Contributing to torch_activation

Thank you for your interest in contributing. We welcome improvements to activation functions, test coverage, and documentation.

## Adding a New Activation Function

### 1. Implementation

Implement your activation as a class inheriting from `BaseActivation` in the appropriate module file (`classical/` or `adaptive/`). Decorate it with `@register_activation`:

```python
from torch_activation.base import BaseActivation
from torch_activation import register_activation

@register_activation
class MyActivation(BaseActivation):
    '''
    Brief description. Define as:
    
    :math:`y = f(x)`

    Args:
        param_a (float, optional): Parameter description. Default: ``1.0``
        inplace (bool, optional): If True, operation is in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    References:
        - Author et al., "Paper Title," Conference/Journal, Year.

    Examples::
        >>> m = MyActivation()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
    '''
    def __init__(self, param_a: float = 1.0, inplace=False, **kwargs):
        super().__init__(inplace, **kwargs)
        self.param_a = param_a

    def _forward(self, x):
        return x * self.param_a

    def _forward_inplace(self, x):
        x *= self.param_a
        return x
```

### 2. Required Docstring Sections

- `:math:` formula (enforced by `scripts/check_docstrings.py`)
- `Args` with parameter descriptions
- `Shape` with input/output dimensions
- `References` with paper citations
- `Examples` with usage

### 3. Testing

Copy `test/_TEMPLATE.py` to `test/test_myactivation.py` and implement the `scalar_ref()` function with the paper formula:

```python
ACTIVATION_NAME = "MyActivation"

def scalar_ref(x: float) -> float:
    """Reference implementation from paper."""
    return x * 1.0  # Your formula here
```

Standard property tests (shape, dtype, gradients, eval determinism, parameter gradients, numerical stability, inplace consistency) run automatically for all registered activations via `test/test_standard_properties.py`.

For non-smooth activations, add the name to `NONSMOOTH_ACTIVATIONS`:

```python
NONSMOOTH_ACTIVATIONS = ["MyActivation"]
```

Run tests locally:

```bash
poetry install --with dev
poetry run pytest test/ -q
```

## Development Setup

```bash
poetry install --with dev
pre-commit install
```

## Pre-commit Hooks

Hooks run automatically before commits:

```bash
pre-commit install
```

Manual run:

```bash
pre-commit run --all-files
```

Hooks enforce:
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Documentation validation (doc8)
- Docstring formula checks

## Commit Style

Use conventional commits with no adjectives or em dashes:

```
type(scope): description

# Examples:
fix(ReLU): correct gradient computation in forward pass
feat(adaptive): add learnable temperature parameter
docs: update activation explorer deployment guide
test: add edge case coverage for numerical stability
chore: update pre-commit dependencies
```

Types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`
