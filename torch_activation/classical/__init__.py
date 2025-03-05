import os
import importlib
import inspect
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get all Python files in the directory (excluding __init__.py)
python_files = [f[:-3] for f in os.listdir(current_dir) 
                if f.endswith('.py') and f != '__init__.py']

# Dictionary to store all classes
__all__ = []

# Import all classes from each file
for module_name in python_files:
    # Import the module
    module = importlib.import_module(f"torch_activation.classical.{module_name}")
    
    # Get all classes from the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if the class is defined in this module (not imported)
        if obj.__module__ == f"torch_activation.classical.{module_name}":
            # Add the class to the current namespace
            globals()[name] = obj
            __all__.append(name)

# Clean up namespace
del os, importlib, inspect, sys, current_dir, python_files, module_name, name, obj, module
