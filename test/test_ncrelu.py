import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["NCReLU"]

ACTIVATION_NAME = "NCReLU"


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, 3)
        out = m(x)
        assert out.shape == (8, 3)

    def test_shape_3d(self):
        m = _get_module()
        x = torch.randn(2, 3, 8)
        out = m(x)
        assert out.shape == (4, 3, 8)


class TestNumerical:
    def test_positive_half(self):
        m = _get_module()
        x = torch.tensor([[1.0, -2.0, 3.0]])
        out = m(x)
        expected_relu = torch.tensor([[1.0, 0.0, 3.0]])
        expected_neg_relu = torch.tensor([[0.0, -2.0, 0.0]])
        expected = torch.cat([expected_relu, expected_neg_relu], dim=0)
        assert torch.allclose(out, expected, atol=1e-6)


class TestGradients:
    def test_finite_diff_nonsmooth(self):
        if ACTIVATION_NAME not in NONSMOOTH_ACTIVATIONS:
            pytest.skip("smooth activation — gradcheck used instead")
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(-2, 2, 6).double().reshape(2, 3)
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_det = x.detach()
        fd = torch.zeros_like(x_det)
        for i in range(x_det.shape[0]):
            for j in range(x_det.shape[1]):
                xp = x_det.clone()
                xp[i, j] += eps
                xm = x_det.clone()
                xm[i, j] -= eps
                fd[i, j] = (m(xp.float()).double().sum() - m(xm.float()).double().sum()) / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-3)

    def test_gradcheck(self):
        if ACTIVATION_NAME in NONSMOOTH_ACTIVATIONS:
            pytest.skip(f"{ACTIVATION_NAME} is non-smooth; using finite-diff check instead")


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.randn(2, 4)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
