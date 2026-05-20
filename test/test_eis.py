import math
import pytest
import torch
import torch_activation

ACTIVATION_NAME = "EIS"


def _get_module(**kw):
    return torch_activation.EIS(**kw)


class TestShape:
    def test_shape_4d(self):
        assert _get_module()(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)

    def test_shape_1d(self):
        assert _get_module()(torch.randn(16)).shape == (16,)


class TestNumerical:
    def test_manual(self):
        m = _get_module()
        with torch.no_grad():
            m.d.fill_(1.0)
            m.e.fill_(1.0)
        x = torch.tensor([0.0])
        expected = torch.tensor([0.0])
        assert torch.allclose(m(x), expected, atol=1e-5)

    def test_positive_large(self):
        m = _get_module()
        with torch.no_grad():
            m.d.fill_(1.0)
            m.e.fill_(1.0)
        x = torch.tensor([10.0])
        expected = x / (1 + torch.exp(-x))
        assert torch.allclose(m(x), expected, atol=1e-4)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        m_d = m.double()
        assert torch.autograd.gradcheck(m_d, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        out = m(torch.randn(4, 4))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
