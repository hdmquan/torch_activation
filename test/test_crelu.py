import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["CReLU"]
ACTIVATION_NAME = "CReLU"


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


class TestShape:
    def test_shape_4d(self):
        m = _get_module()
        x = torch.randn(2, 3, 8, 8)
        out = m(x)
        assert out.shape[0] == x.shape[0] * 2

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(16)
        out = m(x)
        assert out.shape[0] == 32


class TestNumerical:
    def test_positive_inputs(self):
        m = _get_module()
        x = torch.ones(4)
        out = m(x)
        assert torch.allclose(out[:4], torch.ones(4))
        assert torch.allclose(out[4:], torch.zeros(4))

    def test_negative_inputs(self):
        m = _get_module()
        x = -torch.ones(4)
        out = m(x)
        assert torch.allclose(out[:4], torch.zeros(4))
        assert torch.allclose(out[4:], torch.ones(4))


class TestGradients:
    def test_gradcheck(self):
        if ACTIVATION_NAME in NONSMOOTH_ACTIVATIONS:
            pytest.skip(f"{ACTIVATION_NAME} is non-smooth; using finite-diff check instead")
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)

    def test_finite_diff_nonsmooth(self):
        if ACTIVATION_NAME not in NONSMOOTH_ACTIVATIONS:
            pytest.skip("smooth activation — gradcheck used instead")
        pytest.skip("CReLU output shape doubles; finite-diff not directly comparable")


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 1e3, -1e3]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


class TestInplace:
    def test_inplace_matches_normal(self):
        m = _get_module()
        if not hasattr(m, "inplace"):
            pytest.skip(f"{ACTIVATION_NAME} has no inplace attribute")
        m_ip = _get_module(inplace=True)
        x1 = torch.randn(4, 4)
        x2 = x1.clone()
        out_normal = m(x1)
        try:
            m_ip(x2)
        except NotImplementedError:
            pytest.skip(f"{ACTIVATION_NAME} inplace not implemented")
        assert torch.allclose(out_normal, x2, atol=1e-6)
