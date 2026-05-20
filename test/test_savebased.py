import math
import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = []
ACTIVATION_NAME = "SAVEBased"


def scalar_ref(x: float) -> float:
    return x + 0.5 * math.sin(x)


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

    def test_shape_scalar(self):
        m = _get_module()
        x = torch.randn(())
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(16)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_allclose_ref_default(self):
        m = _get_module(alpha=1.0, beta=0.5)
        x = torch.linspace(-3, 3, 50)
        expected = _ref_tensor(x)
        assert torch.allclose(m(x), expected, atol=1e-5)

    def test_zeros(self):
        m = _get_module()
        x = torch.zeros(8)
        out = m(x)
        assert torch.allclose(out, torch.zeros(8), atol=1e-6)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 1e3, -1e3]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
