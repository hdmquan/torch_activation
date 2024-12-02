import torch
from torch_activation import *  # Import all activations and layers from your library
from utils import (
    check_forward_pass,
    check_backward_pass,
)  # Assuming these are defined elsewhere

differentiable_activations = [
    "ShiLU",
    "DELU",
    "CReLU",
    "GCU",
    "CosLU",
    "CoLU",
    "ReLUN",
    "SquaredReLU",
    "ScaledSoftSign",
    "ReGLU",
    "GeGLU",
    "SeGLU",
    "SwiGLU",
    "SinLU",
    "Phish",
    "StarReLU",
]

non_differentiable_activations = [
    # "plot_activation"
]


def test_differentiable_activations(activation_functions, device="cpu"):
    for activation_name in activation_functions:
        activation_fn = globals().get(activation_name)

        # Test the test :)
        # activation_fn = torch.nn.ReLU()

        if activation_fn is None:
            print(f"Activation function {activation_name} not found.")
            continue

        print(f"Testing differentiable activation: {activation_name}")
        input_tensor = torch.randn(2, 3).to(device)

        # Test forward pass
        assert check_forward_pass(
            activation_fn, input_tensor, device
        ), f"{activation_name} failed forward pass."

        # Test backward pass
        assert check_backward_pass(
            activation_fn, input_tensor, device
        ), f"{activation_name} failed backward pass."


def test_non_differentiable_activations(activation_functions, device="cpu"):
    for activation_name in activation_functions:
        activation_fn = globals().get(activation_name)

        if activation_fn is None:
            print(f"Activation function {activation_name} not found.")
            continue

        print(f"Testing non-differentiable activation: {activation_name}")
        input_tensor = torch.randn(2, 3).to(device)

        # Test forward pass
        assert check_forward_pass(
            activation_fn, input_tensor, device
        ), f"{activation_name} failed forward pass."


def test_all_activations():
    device = "cpu"
    test_differentiable_activations(differentiable_activations, device)
    test_non_differentiable_activations(non_differentiable_activations, device)


test_all_activations()
