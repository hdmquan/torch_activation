import torch
import torch_activation
from utils import (
    check_forward_pass,
    check_backward_pass,
)

from loguru import logger

# Get all registered activations
diff_acts = torch_activation.get_all_activations(differentiable_only=True)
non_diff_acts = torch_activation.get_all_activations(differentiable_only=False)

def test_diff_acts(acts, dev="cpu"):
    passed_tests = 0
    failed_tests = 0

    for act_name in acts:
        try:
            # Get the class from the registry
            act_class = torch_activation._ACTIVATIONS[act_name]["class"]
            act_fn = act_class()
        except (KeyError, TypeError):
            logger.error(f"Activation function {act_name} not found or couldn't be instantiated.")
            continue

        # logger.debug(act_fn)

        if act_fn is None:
            logger.error(f"Activation function {act_name} not found.")
            continue

        # logger.info(f"Testing differentiable activation: {act_name}")

        # Test forward pass
        if not check_forward_pass(act_fn, dev):
            logger.error(f"{act_name} failed forward pass.")
            failed_tests += 1
        else:
            passed_tests += 1

        # Test backward pass
        if not check_backward_pass(act_fn, dev):
            logger.error(f"{act_name} failed backward pass.")
            failed_tests += 1
        else:
            passed_tests += 1

    return passed_tests, failed_tests


def test_non_diff_acts(acts, dev="cpu"):
    passed_tests = 0
    failed_tests = 0

    for act_name in acts:
        try:
            # Get the class from the registry
            act_class = torch_activation._ACTIVATIONS[act_name]["class"]
            act_fn = act_class()
        except (KeyError, TypeError):
            logger.error(f"Activation function {act_name} not found or couldn't be instantiated.")
            continue

        logger.info(f"Testing non-differentiable activation: {act_name}")
        inp_tensor = torch.randn(2, 3).to(dev)

        # Test forward pass
        if not check_forward_pass(act_fn, inp_tensor, dev):
            logger.error(f"{act_name} failed forward pass.")
            failed_tests += 1
        else:
            passed_tests += 1

    return passed_tests, failed_tests


def test_all_acts():
    dev = "cpu"
    diff_passed, diff_failed = test_diff_acts(diff_acts, dev)
    non_diff_passed, non_diff_failed = test_non_diff_acts(non_diff_acts, dev)

    total_passed = diff_passed + non_diff_passed
    total_failed = diff_failed + non_diff_failed

    logger.info(
        f"\033[92mSummary: {total_passed} tests passed\033[0m, \033[91m{total_failed} tests failed\033[0m."
    )


test_all_acts()
