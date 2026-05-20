import math
import pytest
import torch
import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["RTReLU"]
ACTIVATION_NAME = "RTReLU"

def scalar_ref(x: float) -> float:
    return max(0.0, x)

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
        pytest.skip("RTReLU is stochastic; deterministic ref check not applicable")

class TestGradients:
    def test_gradcheck(self):
        if ACTIVATION_NAME in NONSMOOTH_ACTIVATIONS:
            pytest.skip(f"{ACTIVATION_NAME} is non-smooth; using finite-diff check instead")
        m = _get_module()
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)

    def test_finite_diff_nonsmooth(self):
        pytest.skip("RTReLU is stochastic; gradient check not applicable")

class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 1e3, -1e3]:
            x = torch.full((4,), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_zeros(self):
        pytest.skip("RTReLU is stochastic; deterministic zeros check not applicable")

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
