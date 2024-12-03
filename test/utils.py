import torch
from loguru import logger


def check_forward_pass(act_fn, device="cpu"):
    try:
        inp = torch.rand(3, 3).to(device)
        _ = act_fn(inp)

        # logger.debug(_.shape)

        return True
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return False


def check_backward_pass(act_fn, device="cpu"):
    try:
        inp = torch.rand(3, 3).to(device)
        inp.requires_grad = True
        output = act_fn(inp)
        output.sum().backward()
        return inp.grad is not None
    except Exception as e:
        logger.error(f"Backward pass failed: {e}")
        return False


def check_gradient(act_fn, epsilon=1e-5, device="cpu"):
    inp = inp.to(device)
    inp.requires_grad = True

    # Compute gradient using autograd
    output = act_fn(inp)
    output.sum().backward()
    grad_autograd = inp.grad.clone()

    # Compute gradient using finite differences
    inp_1 = inp.clone().detach().requires_grad_(False)
    inp_2 = inp_1.clone()

    inp_1.data += epsilon
    output_1 = act_fn(inp_1)

    inp_2.data -= epsilon
    output_2 = act_fn(inp_2)

    grad_fd = (output_1 - output_2).sum() / (2 * epsilon)

    # Compare gradients
    return torch.allclose(grad_autograd, grad_fd, atol=1e-4)
