import pytest
import torch

import torch_activation

NONSMOOTH_ACTIVATIONS: list[str] = ["RReLU"]
ACTIVATION_NAME = "RReLU"


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
        pytest.skip("RReLU is stochastic; deterministic ref check not applicable")


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("RReLU is non-smooth and stochastic")

    def test_finite_diff_nonsmooth(self):
        pytest.skip("RReLU is stochastic")


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
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestInplace:
    def test_inplace_matches_normal(self):
        pytest.skip("RReLU is stochastic; inplace and normal sample independent slopes")
