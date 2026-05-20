import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["DualReLU"]

ACTIVATION_NAME = "DualReLU"


def _get_module(**kwargs):
    return torch_activation.DualReLU(**kwargs)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, 2)
        out = m(x)
        assert out.shape == (4, 1)

    def test_shape_3d(self):
        m = _get_module()
        x = torch.randn(3, 2, 8)
        out = m(x)
        assert out.shape == (3, 1, 8)


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [2.0, -1.0], [-1.0, -2.0]])
        out = m(x)
        expected = torch.tensor([[1.0 - 2.0], [0.0 - 3.0], [2.0 - 0.0], [0.0 - 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)


class TestGradients:
    def test_finite_diff_nonsmooth(self):
        m = _get_module()
        eps = 1e-4
        x = torch.randn(4, 2).double()
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_det = x.detach()
        fd = torch.zeros_like(x_det)
        for i in range(x_det.shape[0]):
            for j in range(x_det.shape[1]):
                xp = x_det.clone(); xp[i, j] += eps
                xm = x_det.clone(); xm[i, j] -= eps
                fd[i, j] = (m(xp.float()).double().sum() - m(xm.float()).double().sum()) / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=5e-3)

    def test_gradcheck(self):
        pytest.skip("DualReLU is non-smooth; using finite-diff check instead")


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.randn(4, 2)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
