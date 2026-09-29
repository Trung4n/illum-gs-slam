"""RASTERIZER_PIXEL_OFFSET re-derived from getProjectionMatrix2 and the CUDA
ndc2Pix formula (docs/DECISIONS.md D29). CPU, needs torch."""
import pytest

torch = pytest.importorskip("torch")

from gaussian_splatting.utils.graphics_utils import (  # noqa: E402
    RASTERIZER_PIXEL_OFFSET,
    getProjectionMatrix2,
)


def ndc2pix(v, s):
    # cuda_rasterizer/auxiliary.h
    return ((v + 1.0) * s - 1.0) * 0.5


@pytest.mark.parametrize("x, y, z", [(0.3, -0.2, 1.7), (-1.1, 0.4, 3.2), (0.0, 0.0, 0.9)])
def test_rasterizer_pixel_is_opencv_pixel_minus_offset(x, y, z):
    # getProjectionMatrix2 builds the matrix in float32: ~1e-8 px of round-off,
    # far below the 0.5 px being checked.
    fx, fy, cx, cy, w, h = 300.0, 290.0, 299.5, 169.5, 600, 340
    p = getProjectionMatrix2(0.01, 100.0, cx, cy, fx, fy, w, h).double()
    hom = p @ torch.tensor([x, y, z, 1.0], dtype=torch.float64)
    ndc = hom[:2] / hom[3]
    pix = (ndc2pix(ndc[0].item(), w), ndc2pix(ndc[1].item(), h))
    opencv = (fx * x / z + cx, fy * y / z + cy)
    assert pix[0] == pytest.approx(opencv[0] - RASTERIZER_PIXEL_OFFSET, abs=1e-4)
    assert pix[1] == pytest.approx(opencv[1] - RASTERIZER_PIXEL_OFFSET, abs=1e-4)
