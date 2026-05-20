import math

import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = []
ACTIVATION_NAME = "tSoftmax"


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, 8)
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(16)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_sums_to_one(self):
        m = _get_module()
        x = torch.randn(4, 8)
        out = m(x)
        assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_nonnegative(self):
        m = _get_module()
        x = torch.randn(4, 8)
        out = m(x)
        assert (out >= 0).all()


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.randn(4, 8)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
