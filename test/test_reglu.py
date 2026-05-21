import pytest
import torch
import torch.nn.functional as F

import torch_activation

ACTIVATION_NAME = "ReGLU"


def _get_module(**kw):
    return torch_activation.ReGLU(**kw)


class TestShape:
    def test_split_last(self):
        m = _get_module()
        x = torch.randn(2, 6, 4)
        assert m(x).shape == (2, 6, 2)

    def test_split_2d(self):
        m = _get_module()
        x = torch.randn(4, 2)
        assert m(x).shape == (4, 1)


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.randn(4, 4)
        a, b = x.chunk(2, dim=-1)
        expected = a * F.relu(b)
        assert torch.allclose(m(x), expected, atol=1e-5)


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.randn(2, 4)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
