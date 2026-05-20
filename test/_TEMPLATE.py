"""
Test template for activation functions.
Copy to test/test_<name>.py and fill in the three marked sections.

1. Set ACTIVATION_NAME to the class name.
2. Implement scalar_ref with the paper formula.
3. Add the name to NONSMOOTH_ACTIVATIONS if the activation has discontinuous derivatives.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = []

ACTIVATION_NAME = "ActivationName"


def scalar_ref(x: float) -> float:
    raise NotImplementedError


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


def _ref_tensor(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1).tolist()
    out = [scalar_ref(v) for v in flat]
    return torch.tensor(out, dtype=x.dtype).reshape(x.shape)


class TestShape:
    def test_shape_4d(self):
        m = _get_module()
        x = torch.randn(2, 3, 8, 8)
        assert m(x).shape == x.shape

    def test_shape_scalar(self):
        m = _get_module()
        x = torch.randn(())
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(16)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 50)
        expected = _ref_tensor(x)
        assert torch.allclose(
            m(x), expected, atol=1e-5
        ), f"max error: {(m(x) - expected).abs().max().item()}"


class TestGradients:
    def test_gradcheck(self):
        if ACTIVATION_NAME in NONSMOOTH_ACTIVATIONS:
            pytest.skip("non-smooth: use finite diff test")
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)

    def test_finite_diff_nonsmooth(self):
        if ACTIVATION_NAME not in NONSMOOTH_ACTIVATIONS:
            pytest.skip("smooth: use gradcheck test")
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(-2, 2, 20).double()
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_np = x.detach()
        fd = (m((x_np + eps).float()) - m((x_np - eps).float())).double() / (2 * eps)
        assert torch.allclose(
            grad_auto, fd, atol=1e-3
        ), f"finite diff mismatch: max {(grad_auto - fd).abs().max().item()}"


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 1e3, -1e3]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any(), f"NaN for x={val}"
            assert not torch.isinf(out).any(), f"Inf for x={val}"

    def test_zeros(self):
        m = _get_module()
        x = torch.zeros(8)
        out = m(x)
        assert torch.allclose(out, torch.full_like(out, scalar_ref(0.0)), atol=1e-6)


class TestInplace:
    def test_inplace_matches_normal(self):
        m = _get_module()
        if not hasattr(m, "inplace"):
            pytest.skip("no inplace support")
        m_ip = _get_module(inplace=True)
        x1 = torch.randn(4, 4)
        x2 = x1.clone()
        out_normal = m(x1)
        m_ip(x2)
        assert torch.allclose(out_normal, x2, atol=1e-6)
