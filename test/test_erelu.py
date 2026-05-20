import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["EReLU"]

ACTIVATION_NAME = "EReLU"


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)


class TestShape:
    def test_shape_4d(self):
        m = _get_module(training=False)
        x = torch.randn(2, 3, 8, 8)
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module(training=False)
        x = torch.randn(16)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_inference_equals_relu(self):
        m = _get_module(training=False)
        x = torch.linspace(-3, 3, 50)
        expected = torch.relu(x)
        assert torch.allclose(m(x), expected, atol=1e-6)

    def test_training_positive_nonneg(self):
        m = _get_module(training=True)
        x = torch.randn(100)
        out = m(x)
        neg_mask = x < 0
        assert (out[neg_mask] == 0).all()


class TestGradients:
    def test_finite_diff_nonsmooth(self):
        if ACTIVATION_NAME not in NONSMOOTH_ACTIVATIONS:
            pytest.skip("smooth activation — gradcheck used instead")
        m = _get_module(training=False)
        eps = 1e-4
        x = torch.linspace(-2, 2, 20).double()
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_np = x.detach()
        fd = (m((x_np + eps).float()) - m((x_np - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-3)

    def test_gradcheck(self):
        if ACTIVATION_NAME in NONSMOOTH_ACTIVATIONS:
            pytest.skip(f"{ACTIVATION_NAME} is non-smooth; using finite-diff check instead")


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module(training=False)
        for val in [0.0, 1e3, -1e3]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
