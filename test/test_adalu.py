import math

import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["AdaLU"]
ACTIVATION_NAME = "AdaLU"


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


class TestShape:
    def test_shape_4d(self):
        m = _get_module()
        x = torch.randn(2, 3, 8, 8)
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(16)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_piecewise_sign(self):
        m = _get_module()
        with torch.no_grad():
            m.alpha.fill_(1.0)
            m.beta.fill_(1.0)
        x = torch.linspace(-3, 3, 50)
        out = m(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_zeros(self):
        m = _get_module()
        x = torch.zeros(8)
        out = m(x)
        assert torch.allclose(out, torch.zeros(8), atol=1e-6)


class TestGradients:
    def test_finite_diff(self):
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(-2, 2, 20).double()
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_np = x.detach()
        fd = (m((x_np + eps).float()) - m((x_np - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-2)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 5.0, -5.0]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
