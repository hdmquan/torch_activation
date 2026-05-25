import pytest

import torch_activation

SKIP_ALL = {"BaseActivation"}


def get_all_names():
    return [n for n in torch_activation._ACTIVATIONS.keys() if n not in SKIP_ALL]


@pytest.mark.parametrize("name", get_all_names())
def test_has_formula(name):
    cls = torch_activation._ACTIVATIONS[name]["class"]
    doc = cls.__doc__ or ""
    if ":math:" not in doc and ".. math::" not in doc:
        pytest.fail(f"{name}: docstring missing math formula (:math: or .. math::)")


@pytest.mark.parametrize("name", get_all_names())
def test_has_shape_section(name):
    cls = torch_activation._ACTIVATIONS[name]["class"]
    doc = cls.__doc__ or ""
    if "Shape:" not in doc:
        pytest.fail(f"{name}: docstring missing 'Shape:' section")


@pytest.mark.parametrize("name", get_all_names())
def test_has_examples_section(name):
    cls = torch_activation._ACTIVATIONS[name]["class"]
    doc = cls.__doc__ or ""
    if "Examples::" not in doc:
        pytest.fail(f"{name}: docstring missing 'Examples::' section")


@pytest.mark.parametrize("name", get_all_names())
def test_has_image_ref(name):
    cls = torch_activation._ACTIVATIONS[name]["class"]
    doc = cls.__doc__ or ""
    expected = f".. image:: ../images/activation_images/{cls.__name__}.png"
    if expected not in doc:
        pytest.fail(f"{name}: docstring missing image reference '{expected}'")
