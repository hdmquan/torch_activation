import os
import importlib
import inspect

__version__ = "0.3.9"

__all__ = []

current_dir = os.path.dirname(__file__)

# Registry to store all activation functions
_ACTIVATIONS = {}


def register_activation(cls=None, *, differentiable=True):
    """
    Decorator to register activation functions.

    Args:
        cls: The class to register
        differentiable: Whether the activation is differentiable
    """

    def _register(cls):
        name = cls.__name__
        _ACTIVATIONS[name] = {"class": cls, "differentiable": differentiable}
        # Also make the class available at module level
        globals()[name] = cls
        __all__.append(name)
        return cls

    if cls is None:
        return _register
    return _register(cls)


# Function to get all registered activations
def get_all_activations(differentiable_only=None):
    """
    Get all registered activation functions.

    Args:
        differentiable_only: If True, return only differentiable activations.
                            If False, return only non-differentiable activations.
                            If None, return all activations.

    Returns:
        List of activation function names
    """
    if differentiable_only is None:
        return list(_ACTIVATIONS.keys())
    return [
        name
        for name, info in _ACTIVATIONS.items()
        if info["differentiable"] == differentiable_only
    ]


# Import and register classes from Python files in the main directory
for file_name in os.listdir(current_dir):
    if file_name.endswith(".py") and file_name != "__init__.py":
        # .py
        module_name = file_name[:-3]
        module = importlib.import_module(f".{module_name}", package=__package__)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                # Register the class with the activation registry
                register_activation(obj)

# Import and conditionally register classes from the classical subdirectory
try:
    classical_module = importlib.import_module(".classical", package=__package__)
    adaptive_module = importlib.import_module(".adaptive", package=__package__)

    # Get all classes from the classical module
    for name in getattr(classical_module, "__all__", []):
        # Get the class object
        cls = getattr(classical_module, name)

        # Make it available at the top level without registering
        globals()[name] = cls
        __all__.append(name)

    # Get all classes from the adaptive module
    for name in getattr(adaptive_module, "__all__", []):
        # Get the class object
        cls = getattr(adaptive_module, name)


except ImportError:
    pass
