import torch
import torch_activation
from utils import (
    check_forward_pass,
    check_backward_pass,
)

from loguru import logger

diff_acts = [
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

non_diff_acts = [
    # "plot_activation"
]


def test_diff_acts(acts, dev="cpu"):
    for act_name in acts:
        act_fn = getattr(torch_activation, act_name, None)()

        logger.debug(act_fn)

        if act_fn is None:
            logger.error(f"Activation function {act_name} not found.")
            continue

        logger.info(f"Testing differentiable activation: {act_name}")

        # Test forward pass
        if not check_forward_pass(act_fn, dev):
            logger.error(f"{act_name} failed forward pass.")

        # Test backward pass
        if not check_backward_pass(act_fn, dev):
            logger.error(f"{act_name} failed backward pass.")


def test_non_diff_acts(acts, dev="cpu"):
    for act_name in acts:
        act_fn = globals().get(act_name)

        if act_fn is None:
            logger.error(f"Activation function {act_name} not found.")
            continue

        logger.info(f"Testing non-differentiable activation: {act_name}")
        inp_tensor = torch.randn(2, 3).to(dev)

        # Test forward pass
        if not check_forward_pass(act_fn, inp_tensor, dev):
            logger.error(f"{act_name} failed forward pass.")


def test_all_acts():
    dev = "cpu"
    test_diff_acts(diff_acts, dev)
    test_non_diff_acts(non_diff_acts, dev)


test_all_acts()
