import os
import importlib
import inspect


__all__ = []

current_dir = os.path.dirname(__file__)

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

__version__ = "0.2.0"
