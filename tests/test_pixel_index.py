"""gaussian_splatting/utils/pixel_index.py against Open3D's real
back-projection (needs open3d; no CUDA)."""
import numpy as np
import pytest

from gaussian_splatting.utils.pixel_index import decode_pixel_index, pixel_index_image


def test_encode_decode_roundtrip_without_open3d():
    code = pixel_index_image(340, 600)
    colors = code.reshape(-1, 3) / 255.0
    np.testing.assert_array_equal(decode_pixel_index(colors), np.arange(340 * 600))


def test_rejects_colors_that_are_not_codes():
    with pytest.raises(RuntimeError):
        decode_pixel_index(np.array([[0.5, 0.2, 0.1]]))


def test_too_many_pixels():
    with pytest.raises(ValueError):
        pixel_index_image(4097, 4097)


def test_open3d_points_carry_their_pixel():
    # Same Open3D calls as GaussianModel.create_pcd_from_image_and_depth, on
    # a depth map with the awkward values a noisy monocular guess can hold
    # (tiny, large, holes) and a non-trivial pose. Every returned point must
    # be exactly the back-projection of the pixel its code names.
    o3d = pytest.importorskip("open3d")
    h, w = 340, 600
    fx = fy = 300.0
    cx, cy = 299.5, 169.5
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.3, 6.0, (h, w)).astype(np.float32)
    depth[rng.random((h, w)) < 0.1] = 0.0          # no Gaussian there
    depth[rng.random((h, w)) < 0.01] = 1e-6        # nearly at the camera
    depth[rng.random((h, w)) < 0.01] = 99.0        # just below depth_trunc
    a = 0.8
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    w2c[:3, 3] = [2.5, -1.2, 4.0]

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(pixel_index_image(h, w)),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=100.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy),
        extrinsic=w2c,
        project_valid_depth_only=True,
    ).random_down_sample(0.25)

    idx = decode_pixel_index(np.asarray(pcd.colors))
    v, u = idx // w, idx % w
    z = depth[v, u].astype(np.float64)
    assert (z > 0).all()
    cam_pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)], 1)
    world = cam_pts @ np.linalg.inv(w2c.astype(np.float64)).T
    np.testing.assert_allclose(np.asarray(pcd.points), world[:, :3], rtol=1e-12, atol=1e-12)
