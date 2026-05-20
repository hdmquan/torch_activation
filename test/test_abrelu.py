import pytest
import torch
import torch.nn.functional as F

import torch_activation

ACTIVATION_NAME = "ABReLU"


def _get_module(**kw):
    return torch_activation.ABReLU(**kw)


class TestShape:
    def test_shape_4d(self):
        assert _get_module()(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)

    def test_shape_1d(self):
        assert _get_module()(torch.randn(16)).shape == (16,)


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 50)
        expected = F.relu(x - x.mean())
        assert torch.allclose(m(x), expected, atol=1e-5)

    def test_zero_mean_input(self):
        m = _get_module()
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected = F.relu(x - x.mean())
        assert torch.allclose(m(x), expected, atol=1e-5)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        out = m(torch.randn(4, 4))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
