"""light_models/normals.py: depth_fd with a configurable stencil (D40)."""
import pytest

torch = pytest.importorskip("torch")

from light_models.normals import DepthFiniteDifference  # noqa: E402


def _plane(h=30, w=40, noise=0.0, seed=0):
    # Points of the plane z = 2 + 0.3 x - 0.2 y seen by a pinhole camera.
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    xn, yn = (u - w / 2) / 50.0, (v - h / 2) / 50.0
    z = 2.0 / (1.0 - 0.3 * xn + 0.2 * yn)
    if noise:
        z = z + noise * torch.randn(z.shape, generator=torch.Generator().manual_seed(seed), dtype=z.dtype)
    pts = torch.stack([xn * z, yn * z, z])
    n_true = torch.tensor([0.3, -0.2, -1.0], dtype=torch.float64)
    return pts, n_true / n_true.norm()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_plane_normal_exact_for_any_stencil(k):
    pts, n_true = _plane()
    n, ok = DepthFiniteDifference(stencil_px=k)(pts, torch.ones(pts.shape[1:], dtype=torch.bool), None)
    torch.testing.assert_close(n[:, ok], n_true[:, None].expand(3, int(ok.sum())))
    # A border of k pixels has no central difference.
    assert not ok[:k].any() and not ok[:, -k:].any() and ok[k:-k, k:-k].all()


def test_wider_stencil_averages_out_bumps():
    pts, n_true = _plane(noise=0.002)
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)

    def mean_angle(k):
        n, ok = DepthFiniteDifference(stencil_px=k)(pts, valid, None)
        c = (n[:, ok] * n_true[:, None]).sum(0).clamp(-1, 1)
        return torch.rad2deg(torch.acos(c)).mean()

    assert mean_angle(4) < 0.5 * mean_angle(1)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "2"])
def test_stencil_validated(k):
    with pytest.raises(ValueError, match="stencil_px"):
        DepthFiniteDifference(stencil_px=k)
