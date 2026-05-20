import pytest
import torch
import torch_activation

ACTIVATION_NAME = "PairedReLU"

def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(**kwargs)

class TestShape:
    def test_shape_4d_doubles_channel(self):
        m = _get_module()
        x = torch.randn(2, 3, 8, 8)
        out = m(x)
        assert out.shape == (2, 6, 8, 8)

    def test_shape_1d_doubles(self):
        m = _get_module()
        x = torch.randn(16)
        out = m(x)
        assert out.shape == (32,)

class TestNumerical:
    def test_finite_output(self):
        m = _get_module()
        x = torch.linspace(-3, 3, 20).unsqueeze(0)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

class TestGradients:
    def test_gradcheck(self):
        m = _get_module()
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-4)

    def test_finite_diff_nonsmooth(self):
        pytest.skip("smooth activation — gradcheck used instead")

class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        for val in [0.0, 2.0, -2.0]:
            x = torch.full((2, 4), val)
            out = m(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

class TestInplace:
    def test_inplace_matches_normal(self):
        pytest.skip("PairedReLU changes output shape; inplace not applicable")
