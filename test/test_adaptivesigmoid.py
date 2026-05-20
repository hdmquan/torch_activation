import math

import pytest
import torch

import torch_activation

ACTIVATION_NAME = "AdaptiveSigmoid"


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


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
    def test_finite_output_nonzero(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 20)
        x = x[x.abs() > 0.1]
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64) * 0.5 + 1.0
        x.requires_grad_(True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)

    def test_finite_diff_nonsmooth(self):
        pytest.skip("smooth activation — gradcheck used instead")


class TestEdgeCases:
    def test_no_nan_inf_nonzero(self):
        m = _get_module()
        for val in [2.0, -2.0, 1.0, -1.0]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


class TestInplace:
    def test_inplace_matches_normal(self):
        m = _get_module()
        m_ip = _get_module(inplace=True)
        x1 = torch.randn(4, 4) + 1.0
        x2 = x1.clone()
        out_normal = m(x1)
        out_ip = m_ip(x2)
        assert torch.allclose(out_normal, out_ip, atol=1e-6)
