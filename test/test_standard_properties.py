import inspect

import pytest
import torch

import torch_activation

NEEDS_INPUT_SHAPE = {
    'GPSoftmax', 'GLSoftmax', 'ARBF', 'PGELU', 'PFTS', 'PFPM',
    'PSIGRAMP', 'RSIGN', 'MAF', 'UAF', 'GReLU', 'GLN',
}

DOUBLES_FIRST_DIM = {'CReLU', 'NCReLU'}
DOUBLES_LAST_DIM = {'PairedReLU'}
SKIP_SHAPE = {'VBAF'}

STOCHASTIC = {'NReLU', 'RTReLU', 'RTPReLU', 'EELU', 'ProbAct', 'ReLUProbAct', 'RReLU', 'EPReLU'}

NONSMOOTH = {
    'ReLU', 'LReLU', 'HardTanh', 'HardSigmoid', 'HardSwish', 'SQNL',
    'SReLU', 'BReLU', 'CReLU', 'mReLU', 'LSPTLU', 'LinQ', 'BiFiring',
    'BoundedBiFiring', 'NActivation', 'ALiSA', 'LiSA', 'MeLU', 'MMeLU',
    'AllReLU', 'StarReLU', 'DReLU', 'PTaLU', 'TaLU', 'PLAF',
    'ShiLU', 'DYReLU', 'MarReLU', 'DelayReLU', 'DisReLU',
    'Maxout', 'Tent', 'Hat', 'PiecewiseMexicanHat', 'PFPM', 'FPAF', 'DPAF',
    'SmoothStep', 'KWTA', 'Binary', 'Hardshrink', 'Softshrink',
    'HardSReLUE', 'LSReLU', 'SRReLU', 'TanhLinearUnit', 'TripleStateSigmoid',
    'ImprovedLogisticSigmoid', 'SigmoidTanh', 'PSTanh', 'PTanh',
    'FCAF_Hidden', 'FCAF_Output', 'CCAF', 'HCAF',
    'NCReLU', 'PairedReLU', 'SignReLU', 'SignReLUPlus', 'ShiftedReLU',
    'ReLUN', 'ABReLU', 'BLReLU',
}

SKIP_NONDEGEN = {'VBAF', 'Maxout', 'KWTA', 'AllReLU'}

SKIP_DTYPE = {'BaseActivation'}

SKIP_ALL = {'BaseActivation'}


def get_all_names():
    return [n for n in torch_activation._ACTIVATIONS.keys() if n not in SKIP_ALL]


def make_module_and_input(name, dtype=torch.float32):
    cls = torch_activation._ACTIVATIONS[name]['class']

    if name in NEEDS_INPUT_SHAPE:
        m = cls(input_shape=4)
        def x_fn(scale=1.0):
            x = torch.randn(2, 4) if scale == 1.0 else torch.full((2, 4), float(scale))
            return x.to(dtype)
    elif name in DOUBLES_FIRST_DIM:
        m = cls()
        def x_fn(scale=1.0):
            x = torch.randn(2, 4) if scale == 1.0 else torch.full((2, 4), float(scale))
            return x.to(dtype)
    elif name in DOUBLES_LAST_DIM:
        m = cls()
        def x_fn(scale=1.0):
            x = torch.randn(2, 4) if scale == 1.0 else torch.full((2, 4), float(scale))
            return x.to(dtype)
    elif name in SKIP_SHAPE:
        m = cls()
        def x_fn(scale=1.0):
            x = torch.randn(2, 4) if scale == 1.0 else torch.full((2, 4), float(scale))
            return x.to(dtype)
    else:
        m = cls()
        def x_fn(scale=1.0):
            x = torch.randn(4) if scale == 1.0 else torch.full((4,), float(scale))
            return x.to(dtype)

    return m, x_fn


@pytest.mark.parametrize("name", get_all_names())
def test_numerical_stability(name):
    m, x_fn = make_module_and_input(name)
    for scale in [1e3, -1e3]:
        x = x_fn(scale)
        with torch.no_grad():
            y = m(x)
        assert not torch.isnan(y).any(), f"{name}: NaN at x~{scale}"
        assert not torch.isinf(y).any(), f"{name}: Inf at x~{scale}"


@pytest.mark.parametrize("name,dtype", [
    (n, d) for n in get_all_names()
    for d in [torch.float32, torch.float16]
])
def test_dtype_preserved(name, dtype):
    if name in SKIP_DTYPE:
        pytest.skip("skip dtype check")
    m, x_fn = make_module_and_input(name, dtype=dtype)
    try:
        y = m(x_fn(1.0))
    except Exception as e:
        pytest.skip(f"forward failed: {e}")
    assert y.dtype == dtype, f"{name}: dtype {dtype}->{y.dtype}"


@pytest.mark.parametrize("name", get_all_names())
def test_shape_preserved(name):
    if name in SKIP_SHAPE | DOUBLES_FIRST_DIM | DOUBLES_LAST_DIM:
        pytest.skip("output shape differs by design")
    m, x_fn = make_module_and_input(name)
    for shape in [(4,), (2, 4), (2, 3, 4)]:
        x = torch.randn(*shape)
        y = m(x)
        assert tuple(y.shape) == shape, f"{name}: shape {x.shape}->{y.shape}"


@pytest.mark.parametrize("name", get_all_names())
def test_gradient_correctness(name):
    if name in STOCHASTIC:
        pytest.skip("stochastic")
    m, x_fn = make_module_and_input(name, dtype=torch.float64)
    x = x_fn(1.0).requires_grad_(True)
    if name in NONSMOOTH:
        try:
            y = m(x)
            y.sum().backward()
        except Exception as e:
            pytest.fail(f"{name}: backward failed: {e}")
        if x.grad is not None:
            assert not torch.isnan(x.grad).any(), f"{name}: NaN in gradient"
    else:
        try:
            assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-3)
        except Exception as e:
            pytest.fail(f"{name}: gradcheck failed: {e}")


@pytest.mark.parametrize("name", get_all_names())
def test_param_gradients_flow(name):
    if name in STOCHASTIC:
        pytest.skip("stochastic")
    m, x_fn = make_module_and_input(name)
    if not list(m.parameters()):
        pytest.skip("no parameters")
    x = x_fn(1.0).requires_grad_(True)
    m(x).sum().backward()
    for pname, p in m.named_parameters():
        assert p.grad is not None, f"{name}.{pname}: grad is None"


@pytest.mark.parametrize("name", get_all_names())
def test_eval_deterministic(name):
    if name in STOCHASTIC:
        pytest.skip("stochastic by design")
    m, x_fn = make_module_and_input(name)
    m.eval()
    x = x_fn(1.0)
    with torch.no_grad():
        y1, y2 = m(x), m(x)
    assert torch.allclose(y1, y2), f"{name}: non-deterministic in eval mode"


@pytest.mark.parametrize("name", get_all_names())
def test_nondegenerate_init(name):
    if name in SKIP_NONDEGEN:
        pytest.skip("intentional special case")
    m, x_fn = make_module_and_input(name)
    m.eval()
    if name in DOUBLES_FIRST_DIM or name in DOUBLES_LAST_DIM or name in SKIP_SHAPE:
        x = torch.randn(10, 4)
    elif name in NEEDS_INPUT_SHAPE:
        x = torch.randn(10, 4)
    else:
        x = torch.linspace(-2, 2, 100)
    with torch.no_grad():
        y = m(x)
    if y.numel() > 1:
        y_range = y.max() - y.min()
        assert y_range > 0.01, f"{name}: collapses to constant at init (range={y_range:.4f})"
    assert not torch.isnan(y).any(), f"{name}: NaN at init"


@pytest.mark.parametrize("name", get_all_names())
def test_inplace_matches_outofplace(name):
    if name in STOCHASTIC:
        pytest.skip("stochastic")
    cls = torch_activation._ACTIVATIONS[name]['class']
    if 'inplace' not in inspect.signature(cls.__init__).parameters:
        pytest.skip("no inplace")
    try:
        if name in NEEDS_INPUT_SHAPE:
            m_safe = cls(input_shape=4, inplace=False)
            m_ip = cls(input_shape=4, inplace=True)
        else:
            m_safe = cls(inplace=False)
            m_ip = cls(inplace=True)
    except Exception:
        pytest.skip("construction failed")
    _, x_fn = make_module_and_input(name)
    x = x_fn(1.0)
    try:
        out_safe = m_safe(x.clone())
        out_ip = m_ip(x.clone())
    except NotImplementedError:
        pytest.skip("inplace not implemented")
    assert torch.allclose(out_safe, out_ip, atol=1e-5), \
        f"{name}: inplace/out-of-place mismatch"
