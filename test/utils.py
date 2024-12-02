import torch


def check_forward_pass(activation_fn, input_tensor, device="cpu"):
    """Check if the forward pass runs without errors"""
    try:
        input_tensor = input_tensor.to(device)
        output = activation_fn(input_tensor)
        return True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False


def check_backward_pass(activation_fn, input_tensor, device="cpu"):
    """Check if backward pass runs without errors"""
    try:
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        output = activation_fn(input_tensor)
        output.sum().backward()
        return input_tensor.grad is not None
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return False


def check_gradient(activation_fn, input_tensor, epsilon=1e-5, device="cpu"):
    """Gradient check using finite difference method for testing"""
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    # Compute gradient using autograd
    output = activation_fn(input_tensor)
    output.sum().backward()
    grad_autograd = input_tensor.grad.clone()

    # Compute gradient using finite differences
    input_tensor_1 = input_tensor.clone().detach().requires_grad_(False)
    input_tensor_2 = input_tensor_1.clone()

    input_tensor_1.data += epsilon
    output_1 = activation_fn(input_tensor_1)

    input_tensor_2.data -= epsilon
    output_2 = activation_fn(input_tensor_2)

    grad_fd = (output_1 - output_2).sum() / (2 * epsilon)

    # Compare gradients
    return torch.allclose(grad_autograd, grad_fd, atol=1e-4)
