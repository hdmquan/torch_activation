import pytest
import torch
import torch_activation

ACTIVATION_NAME = "VAF"


def _get_module(**kw):
    return torch_activation.VAF(**kw)


class TestShape:
    def test_shape_4d(self):
        assert _get_module()(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)

    def test_shape_1d(self):
        assert _get_module()(torch.randn(16)).shape == (16,)


class TestNumerical:
    def test_output_finite(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 20)
        out = m(x)
        assert torch.isfinite(out).all()


class TestGradients:
    def test_gradcheck(self):
        m = _get_module(J=2)
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        m_d = m.double()
        assert torch.autograd.gradcheck(m_d, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        out = m(torch.randn(4, 4))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
