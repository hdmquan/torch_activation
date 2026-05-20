import math
import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ['LogSQNL']
ACTIVATION_NAME = 'LogSQNL'

def scalar_ref(x: float) -> float:
    if x > 2:
        return 1.0
    elif x >= 0:
        return 0.5*x - x**2/4 + 0.5
    elif x >= -2:
        return 0.5*x + x**2/4 + 0.5
    else:
        return 0.0

def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)

def _ref_tensor(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1).tolist()
    out = [scalar_ref(v) for v in flat]
    return torch.tensor(out, dtype=x.dtype).reshape(x.shape)

class TestShape:
    def test_shape_4d(self):
        m = _get_module()
        x = torch.randn(2, 3, 8, 8)
        assert m(x).shape == x.shape

    def test_shape_scalar(self):
        m = _get_module()
        x = torch.randn(())
        assert m(x).shape == x.shape

    def test_shape_1d(self):
        m = _get_module()
        x = torch.randn(16)
        assert m(x).shape == x.shape

class TestNumerical:
    def test_allclose_ref(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 50)
        expected = _ref_tensor(x)
        assert torch.allclose(m(x), expected, atol=1e-5), \
            f"Max error: {(m(x) - expected).abs().max().item()}"

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
        m = _get_module()
        eps = 1e-4
        x = torch.linspace(-1.9, 1.9, 20).double()
        x.requires_grad_(True)
        out = m(x.float()).double()
        out.sum().backward()
        grad_auto = x.grad.clone()
        x_np = x.detach()
        fd = (m((x_np + eps).float()) - m((x_np - eps).float())).double() / (2 * eps)
        assert torch.allclose(grad_auto, fd, atol=1e-3)

class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 1e3, -1e3]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_zeros(self):
        m = _get_module()
        x = torch.zeros(8)
        out = m(x)
        expected_zero = scalar_ref(0.0)
        assert torch.allclose(out, torch.full_like(out, expected_zero), atol=1e-6)

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
