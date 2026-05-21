import warnings
from typing import Any

__version__ = "1.0.0"

_ACTIVATIONS: dict[str, dict[str, Any]] = {}


def register_activation(cls: Any = None, *, differentiable: bool = True) -> Any:
    def _register(cls: type) -> type:
        name: str = cls.__name__
        _ACTIVATIONS[name] = {"class": cls, "differentiable": differentiable}
        globals()[name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def get_all_activations() -> list[str]:
    return list(_ACTIVATIONS.keys())


def _import_submodule(package_path, override=False):
    import importlib
    import inspect
    import os

    current_dir = os.path.dirname(__file__)
    subdir = os.path.join(current_dir, package_path)
    pkg = f"torch_activation.{package_path}"

    for fname in os.listdir(subdir):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        mod_name = fname[:-3]
        try:
            importlib.import_module(f".{mod_name}", package=pkg)
        except Exception as e:
            warnings.warn(f"Failed to import {pkg}.{mod_name}: {e}")

    try:
        pkg_mod = importlib.import_module(f".{package_path}", package="torch_activation")
        for name in getattr(pkg_mod, "__all__", []):
            obj = getattr(pkg_mod, name, None)
            if obj is not None and inspect.isclass(obj) and (override or name not in globals()):
                globals()[name] = obj
    except Exception as e:
        warnings.warn(f"Failed to import torch_activation.{package_path}: {e}")


_import_submodule("adaptive")
_import_submodule("classical", override=True)

__all__ = list(_ACTIVATIONS.keys()) + ["__version__", "get_all_activations", "register_activation"]
