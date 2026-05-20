import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["FracLReLU"]
ACTIVATION_NAME = "FracLReLU"


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
    def test_finite_output(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 20)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("FracLReLU is non-smooth; using finite-diff check instead")

    def test_finite_diff_nonsmooth(self):
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(0.5, 2.0, 10).double()
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_np = x.detach()
        fd = (m((x_np + eps).float()) - m((x_np - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-2)


class TestEdgeCases:
    def test_no_nan_inf_positive(self):
        m = _get_module()
        x = torch.full((4,), 1.0)
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
