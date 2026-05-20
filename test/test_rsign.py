import pytest
import torch

import torch_activation

ACTIVATION_NAME = "RSIGN"
INPUT_SHAPE = 8


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(input_shape=INPUT_SHAPE, **kwargs)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, INPUT_SHAPE)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_finite_output(self):
        m = _get_module()
        x = torch.randn(4, INPUT_SHAPE)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_binary_output(self):
        m = _get_module()
        x = torch.randn(4, INPUT_SHAPE)
        out = m(x)
        assert torch.all((out == 1) | (out == -1))


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("RSIGN is non-smooth; gradcheck not applicable")

    def test_finite_diff_nonsmooth(self):
        pytest.skip("RSIGN is a sign function; gradient is zero almost everywhere")


class TestEdgeCases:
    def test_no_nan_inf(self):
        m = _get_module()
        x = torch.zeros(2, INPUT_SHAPE)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestInplace:
    def test_inplace_matches_normal(self):
        m = _get_module()
        m_ip = _get_module(inplace=True)
        x1 = torch.randn(4, INPUT_SHAPE)
        x2 = x1.clone()
        out_normal = m(x1)
        try:
            out_ip = m_ip(x2)
        except NotImplementedError:
            pytest.skip("inplace not implemented")
        assert torch.allclose(out_normal, out_ip, atol=1e-6)
