# This file isn't even a test :((
# This file in the nutshell:
# - Get all register activation functions
# - Test if they can be inplace
# - If they are not inplace-able, pass inplace to it, if it doesn't return not implement error, raise an error.
# - If they are inplace-able, pass inplace to it, if the output doesn't match the result from the not inplace, error. 
# - If it raise any error that is something other than not implement or if the flag is not implemented, give a warning. else mark it as pass.

import torch
from loguru import logger
import torch_activation as tac
from torch_activation.utils import can_be_inplace


if __name__ == "__main__":

    acts = tac.get_all_activations()
    x = torch.randn(4, 3, 32, 32)

    passed = 0
    warnings = 0
    failed = 0

    for act_name in acts:
        try:
            act_class = tac._ACTIVATIONS[act_name]["class"]
            act_fn = act_class()
        except:
            logger.error(f"{act_name}: Not found in registry")
            failed += 1
            continue
        
        # Is it even a word?
        is_inplaceable = can_be_inplace(act_fn, x)
        
        # Test non-inplace version
        try:
            normal_output = act_fn(x.clone())
        except Exception as e:
            logger.error(f"{act_name}: Failed to run non-inplace version - {str(e)}")
            failed += 1
            continue

        # Test inplace version
        try:
            inplace_fn = act_class(inplace=True)
            x_inplace = x.clone()
            inplace_output = inplace_fn(x_inplace)

            if not is_inplaceable:
                # Should raised NotImplementedError here
                logger.error(f"{act_name}: Allows inplace but derivative depends on output!")
                failed += 1
            else:
                # Check if outputs match
                if torch.allclose(normal_output, inplace_output, rtol=1e-4, atol=1e-4):
                    logger.success(f"{act_name}: Passed inplace test")
                    passed += 1
                else:
                    logger.error(f"{act_name}: Inplace output doesn't match non-inplace output")
                    failed += 1

        except NotImplementedError:
            if is_inplaceable:
                logger.warning(f"{act_name}: Could be inplace but not implemented")
                warnings += 1
            else:
                logger.success(f"{act_name}: Correctly blocks inplace operation")
                passed += 1
        except AttributeError as e:
            if "inplace" in str(e):
                logger.warning(f"{act_name}: Inplace flag was not implemented")
                warnings += 1
            else:
                logger.error(f"{act_name}: Unexpected attribute error - {str(e)}")
                failed += 1
        except Exception as e:
            if is_inplaceable:
                logger.warning(f"{act_name}: Unexpected error - {str(e)}")
                warnings += 1
            else:
                logger.success(f"{act_name}: Correctly blocks inplace operation")
                passed += 1

    print("\nSummary:")
    print(f"\033[32mPassed: {passed}\033[0m")
    print(f"\033[33mWarnings: {warnings}\033[0m")
    print(f"\033[31mFailed: {failed}\033[0m")

    assert failed == 0, "Failed tests"
        