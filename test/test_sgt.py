import math

import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS = ["SGT"]
ACTIVATION_NAME = "SGT"


def scalar_ref(x: float) -> float:
    if x < 0:
        return 0.1 * (x**1.0)
    else:
        return 1.1 * (x**1.0)


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


def _ref_tensor(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1).tolist()
    return torch.tensor([scalar_ref(v) for v in flat], dtype=x.dtype).reshape(x.shape)


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
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 50)
        expected = _ref_tensor(x)
        assert torch.allclose(
            m(x), expected, atol=1e-5
        ), f"Max error: {(m(x) - expected).abs().max().item()}"


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("SGT is non-smooth at x=0; using finite-diff check instead")

    def test_finite_diff_nonsmooth(self):
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(-2, 2, 20).double()
        x = x[x.abs() > 0.01]
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_np = x.detach()
        fd = (m((x_np + eps).float()) - m((x_np - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-3)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 5.0, -5.0]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


class TestInplace:
    def test_inplace_matches_normal(self):
        m = _get_module()
        m_ip = _get_module(inplace=True)
        x1 = torch.randn(4, 4)
        x2 = x1.clone()
        out_normal = m(x1)
        out_ip = m_ip(x2)
        assert torch.allclose(out_normal, out_ip, atol=1e-6)
