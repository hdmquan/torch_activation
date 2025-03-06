import torch
import torch_activation
import inspect
import importlib
import pkgutil
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
    tested_count = 0
    skipped_count = 0

    for act_name in acts:
        try:
            # Get the class from the registry
            act_class = torch_activation._ACTIVATIONS[act_name]["class"]
            act_fn = act_class()
        except (KeyError, TypeError):
            logger.error(f"Activation function {act_name} not found or couldn't be instantiated.")
            skipped_count += 1
            continue

        # logger.debug(act_fn)

        if act_fn is None:
            logger.error(f"Activation function {act_name} not found.")
            skipped_count += 1
            continue

        # logger.info(f"Testing differentiable activation: {act_name}")
        tested_count += 1

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

    return passed_tests, failed_tests, tested_count, skipped_count


def test_non_diff_acts(acts, dev="cpu"):
    passed_tests = 0
    failed_tests = 0
    tested_count = 0
    skipped_count = 0

    for act_name in acts:
        try:
            # Get the class from the registry
            act_class = torch_activation._ACTIVATIONS[act_name]["class"]
            act_fn = act_class()
        except (KeyError, TypeError):
            logger.error(f"Activation function {act_name} not found or couldn't be instantiated.")
            skipped_count += 1
            continue

        logger.info(f"Testing non-differentiable activation: {act_name}")
        tested_count += 1
        inp_tensor = torch.randn(2, 3).to(dev)

        # Test forward pass
        if not check_forward_pass(act_fn, inp_tensor, dev):
            logger.error(f"{act_name} failed forward pass.")
            failed_tests += 1
        else:
            passed_tests += 1

    return passed_tests, failed_tests, tested_count, skipped_count


def find_unregistered_activations():
    """Find activation classes that aren't registered in the registry."""
    registered_classes = {info["class"] for info in torch_activation._ACTIVATIONS.values()}
    unregistered_classes = []
    
    # Recursively import all submodules in torch_activation
    package = torch_activation
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        try:
            module = importlib.import_module(name)
            
            # Look for classes that might be activation functions
            for name, obj in inspect.getmembers(module):
                # Check if it's a class and not imported from elsewhere
                if (inspect.isclass(obj) and 
                    obj.__module__ == module.__name__ and
                    obj not in registered_classes):
                    
                    # Check if it has forward method (potential activation function)
                    if hasattr(obj, 'forward') and callable(getattr(obj, 'forward')):
                        unregistered_classes.append((obj.__name__, obj.__module__))
        except ImportError as e:
            logger.error(f"Error importing module {name}: {e}")
    
    return unregistered_classes


def test_all_acts():
    dev = "cpu"
    diff_passed, diff_failed, diff_tested, diff_skipped = test_diff_acts(diff_acts, dev)
    non_diff_passed, non_diff_failed, non_diff_tested, non_diff_skipped = test_non_diff_acts(non_diff_acts, dev)

    total_passed = diff_passed + non_diff_passed
    total_failed = diff_failed + non_diff_failed
    total_tested = diff_tested + non_diff_tested
    total_skipped = diff_skipped + non_diff_skipped
    total_activations = len(diff_acts) + len(non_diff_acts)
    
    # Find unregistered activation functions
    unregistered = find_unregistered_activations()

    logger.info(
        f"\033[92mSummary: {total_passed} tests passed\033[0m, \033[91m{total_failed} tests failed\033[0m."
    )
    logger.info(
        f"Activation functions: {total_tested} tested, {total_skipped} skipped, {total_activations} total."
    )
    
    if unregistered:
        logger.warning(f"Found {len(unregistered)} potential activation classes that aren't registered:")
        for class_name, module_name in unregistered:
            logger.warning(f"  {class_name} in {module_name}")


test_all_acts()
