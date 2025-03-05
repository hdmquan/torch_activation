import os
import importlib
import inspect


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
    return [name for name, info in _ACTIVATIONS.items() 
            if info["differentiable"] == differentiable_only]

for file_name in os.listdir(current_dir):
    if file_name.endswith(".py") and file_name != "__init__.py":

        # .py
        module_name = file_name[:-3]

        module = importlib.import_module(f".{module_name}", package=__package__)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:

                __all__.append(name)

                # Enable access to the module from the global scope :(
                globals()[name] = obj

__version__ = "0.2.1"
