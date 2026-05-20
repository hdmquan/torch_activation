import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["OPLU"]

ACTIVATION_NAME = "OPLU"


def _get_module(**kwargs):
    return torch_activation.OPLU(**kwargs)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, 4)
        assert m(x).shape == x.shape

    def test_shape_3d(self):
        m = _get_module()
        x = torch.randn(2, 6, 8)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_pairwise_max_min(self):
        m = _get_module()
        x = torch.tensor([[1.0, 3.0, -2.0, 5.0]])
        out = m(x)
        assert torch.allclose(out, torch.tensor([[3.0, 1.0, 5.0, -2.0]]), atol=1e-6)


class TestGradients:
    def test_finite_diff_nonsmooth(self):
        m = _get_module()
        eps = 1e-4
        x = torch.randn(2, 4).double()
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
        assert torch.allclose(grad_auto, fd, atol=1e-2)

    def test_gradcheck(self):
        pytest.skip("OPLU is non-smooth (max/min); using finite-diff check instead")


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.randn(2, 4)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
