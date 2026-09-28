"""utils/color_space.py must use the same sRGB convention as the dataset repo.

The cross-check against Replica/scripts/physics_check/common.py needs the
path of that repo in the REPLICA_REPO environment variable (and OpenEXR,
which common.py imports); it is skipped otherwise. The other tests only need
torch.

    REPLICA_REPO=/path/to/Replica python -m pytest tests/test_color_space.py
"""
import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.color_space import linear2sRGB, sRGB2Linear  # noqa: E402


def _replica_common():
    root = os.environ.get("REPLICA_REPO")
    if not root:
        pytest.skip("REPLICA_REPO not set")
    path = os.path.join(root, "scripts", "physics_check", "common.py")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    pytest.importorskip("OpenEXR")
    spec = importlib.util.spec_from_file_location("replica_physics_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eotf_matches_dataset_repo_on_every_8bit_code():
    common = _replica_common()
    codes = np.arange(256)
    expected = common.srgb_to_lin(codes)
    got = sRGB2Linear(torch.from_numpy(codes / 255.0)).numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_oetf_matches_dataset_repo_encoding_inside_unit_range():
    common = _replica_common()
    lin = np.linspace(0.0, 1.0, 10001)
    expected = common.lin_to_srgb8(lin)
    got = np.rint(linear2sRGB(torch.from_numpy(lin)).numpy() * 255.0)
    np.testing.assert_array_equal(got, expected)


def test_round_trip():
    x = torch.linspace(0.0, 1.0, 10001, dtype=torch.float64)
    torch.testing.assert_close(sRGB2Linear(linear2sRGB(x)), x, rtol=0, atol=1e-6)
    torch.testing.assert_close(linear2sRGB(sRGB2Linear(x)), x, rtol=0, atol=1e-6)


def test_no_clipping_outside_unit_range():
    # Values above 1 keep increasing (loss gradients survive over-exposure).
    x = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)
    y = linear2sRGB(x)
    assert torch.all(y[1:] > y[:-1])
    assert y[0].item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("fn", [sRGB2Linear, linear2sRGB])
def test_gradients_finite_everywhere(fn):
    x = torch.tensor(
        [-0.5, -1e-6, 0.0, 0.0031308, 0.04045, 0.5, 1.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    fn(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad > 0).all()
