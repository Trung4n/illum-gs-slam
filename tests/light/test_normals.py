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
    n, ok = DepthFiniteDifference(stencil_px=k)(pts, torch.ones(pts.shape[1:], dtype=torch.bool), None, None)
    torch.testing.assert_close(n[:, ok], n_true[:, None].expand(3, int(ok.sum())))
    # A border of k pixels has no central difference.
    assert not ok[:k].any() and not ok[:, -k:].any() and ok[k:-k, k:-k].all()


def test_wider_stencil_averages_out_bumps():
    pts, n_true = _plane(noise=0.002)
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)

    def mean_angle(k):
        n, ok = DepthFiniteDifference(stencil_px=k)(pts, valid, None, None)
        c = (n[:, ok] * n_true[:, None]).sum(0).clamp(-1, 1)
        return torch.rad2deg(torch.acos(c)).mean()

    assert mean_angle(4) < 0.5 * mean_angle(1)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "2"])
def test_stencil_validated(k):
    with pytest.raises(ValueError, match="stencil_px"):
        DepthFiniteDifference(stencil_px=k)


def _sensor_camera(h=30, w=40, noise=0.0):
    # Camera whose sensor depth is the plane of _plane (same intrinsics:
    # x = (u - w/2) / 50 at pixel offset 0, the data's convention).
    from types import SimpleNamespace

    pts, n_true = _plane(h, w, noise)
    cam = SimpleNamespace(fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, depth=pts[2].numpy().copy())
    return cam, pts, n_true


def test_sensor_depth_normals_come_from_the_sensor_not_the_map():
    from light_models.normals import SensorDepthFiniteDifference

    cam, pts, n_true = _sensor_camera()
    src = SensorDepthFiniteDifference(stencil_px=1)
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    # The map's points are garbage (a staircase): the result must not care.
    stairs = pts.clone()
    stairs[2] = torch.round(stairs[2] * 10) / 10
    n, ok = src(stairs, valid, None, cam)
    torch.testing.assert_close(n[:, ok], n_true[:, None].expand(3, int(ok.sum())))
    assert ok[1:-1, 1:-1].all()


def test_sensor_depth_normals_respect_validity_and_missing_readings():
    from light_models.normals import SensorDepthFiniteDifference

    cam, pts, _ = _sensor_camera()
    cam.depth[10, 10] = 0.0  # no sensor reading
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    valid[20, 20] = False  # not trusted by the G-buffer
    _, ok = SensorDepthFiniteDifference(stencil_px=1)(pts, valid, None, cam)
    assert not ok[10, 10] and not ok[10, 11] and not ok[11, 10]
    assert not ok[20, 20] and ok[20, 21]


def test_sensor_depth_normals_need_a_depth_and_cache_per_camera():
    import pickle

    from light_models.normals import SensorDepthFiniteDifference

    src = SensorDepthFiniteDifference(stencil_px=1)
    cam, pts, _ = _sensor_camera()
    valid = torch.ones(pts.shape[1:], dtype=torch.bool)
    n1, _ = src(pts, valid, None, cam)
    # Another camera (another frame) must not reuse the cached normals.
    cam2, _, _ = _sensor_camera()
    cam2.depth = cam2.depth[:, ::-1].copy()
    n2, _ = src(pts, valid, None, cam2)
    assert not torch.equal(n1, n2)
    # Pickled to other processes without its cache.
    assert pickle.loads(pickle.dumps(src))._cache is None
    cam.depth = None
    with pytest.raises(ValueError, match="sensor depth"):
        src(pts, valid, None, cam)
