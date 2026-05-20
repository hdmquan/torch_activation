import pytest
import torch
import torch.nn.functional as F

import torch_activation

ACTIVATION_NAME = "BetaSoftmax"


def _get_module(**kw):
    return torch_activation.BetaSoftmax(**kw)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, 8)
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(8)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module(beta=2.0)
        x = torch.randn(4, 8)
        expected = F.softmax(2.0 * x, dim=-1)
        assert torch.allclose(m(x), expected, atol=1e-5)

    def test_sums_to_one(self):
        m = _get_module()
        x = torch.randn(4, 8)
        assert torch.allclose(m(x).sum(-1), torch.ones(4), atol=1e-5)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module(beta=1.0, trainable=False)
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.randn(4, 8)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
