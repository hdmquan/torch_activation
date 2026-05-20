import math
import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS = ["LADA"]
ACTIVATION_NAME = "LADA"


def scalar_ref(x: float) -> float:
    if x >= 0:
        return math.exp(-x)
    return 0.1 * x


def _get_module(**kw):
    return torch_activation.LADA(**kw)


class TestShape:
    def test_shape_4d(self):
        assert _get_module()(torch.randn(2, 3, 8, 8)).shape == (2, 3, 8, 8)

    def test_shape_1d(self):
        assert _get_module()(torch.randn(16)).shape == (16,)


class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module(a=1.0, b=0.0, c=0.1)
        x = torch.linspace(-3, 3, 50)
        expected = torch.tensor([scalar_ref(v) for v in x.tolist()])
        assert torch.allclose(m(x), expected, atol=1e-5)


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("non-smooth at zero")

    def test_finite_diff_nonsmooth(self):
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(0.1, 2, 20).double()
        x.requires_grad_(True)
        m(x.float()).double().sum().backward()
        grad_auto = x.grad.clone()
        xd = x.detach()
        fd = (m((xd + eps).float()) - m((xd - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-3)


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, -1e3]:
            out = m(torch.full((4,), val))
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
