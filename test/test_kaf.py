import pytest
import torch

import torch_activation

ACTIVATION_NAME = "KAF"


def _get_module(**kw):
    return torch_activation.KAF(**kw)


class TestShape:
    def test_shape_4d(self):
        assert _get_module()(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)

    def test_shape_1d(self):
        assert _get_module()(torch.randn(16)).shape == (16,)


class TestNumerical:
    def test_zero_weights_give_zero(self):
        m = _get_module()
        with torch.no_grad():
            m.a.zero_()
        x = torch.randn(4)
        assert torch.allclose(m(x), torch.zeros_like(x), atol=1e-6)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module(D=5)
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        m_d = m.double()
        assert torch.autograd.gradcheck(m_d, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        out = m(torch.randn(4, 4))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
