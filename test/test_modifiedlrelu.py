import math
import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS = ["ModifiedLReLU"]
ACTIVATION_NAME = "ModifiedLReLU"


def scalar_ref(x: float) -> float:
    a = 0.1
    return x if x + a > 0 else -a * x


def _get_module(**kw):
    return torch_activation.ModifiedLReLU(**kw)


class TestShape:
    def test_shape_4d(self):
        assert _get_module()(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)

    def test_shape_1d(self):
        assert _get_module()(torch.randn(16)).shape == (16,)


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 50)
        expected = torch.tensor([scalar_ref(v) for v in x.tolist()])
        assert torch.allclose(m(x), expected, atol=1e-5)


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("non-smooth")

    def test_finite_diff_nonsmooth(self):
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(-2, 2, 20).double()
        x.requires_grad_(True)
        m(x.float()).double().sum().backward()
        grad_auto = x.grad.clone()
        xd = x.detach()
        fd = (m((xd + eps).float()) - m((xd - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-3)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 1e3, -1e3]:
            out = m(torch.full((4,), val))
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
