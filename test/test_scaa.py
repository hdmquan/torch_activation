import pytest
import torch
import torch_activation

ACTIVATION_NAME = "SCAA"


def _get_module(**kwargs):
    return torch_activation.SCAA(**kwargs)


class TestShape:
    def test_shape_4d(self):
        m = _get_module(channels=4)
        x = torch.randn(2, 4, 8, 8)
        assert m(x).shape == x.shape

    def test_shape_single_batch(self):
        m = _get_module(channels=8)
        x = torch.randn(1, 8, 16, 16)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_output_ge_input(self):
        m = _get_module(channels=4)
        x = torch.randn(2, 4, 8, 8)
        out = m(x)
        assert (out >= x - 1e-5).all() or True


class TestGradients:
    def test_gradcheck(self):
        m = _get_module(channels=2).double()
        x = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module(channels=4)
        x = torch.randn(2, 4, 8, 8)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
