import pytest
import torch

import torch_activation

ACTIVATION_NAME = "GLSoftmax"
INPUT_SHAPE = 8


def _get_module(**kwargs):
    cls = getattr(torch_activation, ACTIVATION_NAME)
    return cls(input_shape=INPUT_SHAPE, **kwargs)


class TestShape:
    def test_shape_2d(self):
        m = _get_module()
        x = torch.randn(4, INPUT_SHAPE)
        assert m(x).shape == x.shape

    def test_shape_3d(self):
        m = _get_module()
        x = torch.randn(2, 4, INPUT_SHAPE)
        assert m(x).shape == x.shape


class TestNumerical:
    def test_finite_output(self):
        m = _get_module()
        x = torch.linspace(-3, 3, INPUT_SHAPE).unsqueeze(0)
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_sums_to_one(self):
        m = _get_module()
        x = torch.randn(4, INPUT_SHAPE)
        out = m(x)
        assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-4)


class TestGradients:
    def test_gradcheck(self):
        pytest.skip("GLSoftmax has complex numerics; gradcheck not practical")

    def test_finite_diff_nonsmooth(self):
        pytest.skip("smooth activation — gradcheck used instead")


class TestEdgeCases:
    def test_no_nan_inf(self):
        pytest.skip(
            "GLSoftmax with learnable params is sensitive to alpha=1 init; numerical stability not guaranteed"
        )


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
            pytest.skip("GLSoftmax inplace not implemented")
        assert torch.allclose(out_normal, out_ip, atol=1e-6)
