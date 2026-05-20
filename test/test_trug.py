import math
import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = []
ACTIVATION_NAME = "TruG"


def scalar_ref(x: float) -> float:
    return math.exp(-x * x / 2.0)


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


def _ref_tensor(x):
    flat = x.reshape(-1).tolist()
    out = [scalar_ref(v) for v in flat]
    return torch.tensor(out, dtype=x.dtype).reshape(x.shape)


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
    def test_sigma_one_matches_gaussian(self):
        m = _get_module(sigma=1.0)
        x = torch.linspace(-3, 3, 50)
        expected = _ref_tensor(x)
        assert torch.allclose(m(x), expected, atol=1e-5)

    def test_always_positive(self):
        m = _get_module()
        x = torch.randn(100)
        out = m(x)
        assert (out > 0).all()

    def test_peak_at_zero(self):
        m = _get_module()
        x = torch.zeros(4)
        out = m(x)
        assert torch.allclose(out, torch.ones(4), atol=1e-5)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 5.0, -5.0]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
