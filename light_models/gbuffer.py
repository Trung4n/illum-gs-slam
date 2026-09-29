# Surface points from the rasterized G-buffer (docs/DECISIONS.md D5, D21).
#
# The rasterizer (black background) returns opacity-weighted sums:
# depth = sum_i w_i z_i, opacity = sum_i w_i. The z-depth of a pixel is
# therefore depth / opacity, and only where opacity is large enough to trust
# it (Light.gbuffer.opacity_thr). Back-projection with the camera
# intrinsics:  x_C = z * K^-1 [u, v, 1]^T  (OpenCV camera frame; z is the
# depth along the optical axis, NOT the distance to the camera or the light).
#
# Pixels below the threshold are not dropped: they are shaded at a fallback
# depth, the median z of the trusted pixels of the same image (detached).
# That keeps shading finite there (dividing a tiny depth by a tiny opacity
# would put the point at the lamp and blow up 1/d^2) while the albedo and
# opacity of those pixels still receive gradient, which mapping needs to grow
# the map into new regions. Their depth receives no gradient.

import torch


def pixel_rays(camera, height, width, device, dtype):
    """(3,H,W) K^-1 [u, v, 1]: u = column, v = row, pixel centers at integers
    (same convention as Open3D's back-projection used to create Gaussians)."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = (u - camera.cx) / camera.fx
    y = (v - camera.cy) / camera.fy
    return torch.stack([x, y, torch.ones_like(x)], dim=0)


def surface_points(gbuffer, camera, opacity_thr):
    """Returns (points (3,H,W) in the camera frame, valid (H,W) bool) or
    (None, valid) if no pixel has any opacity at all."""
    depth = gbuffer["depth"][0]
    opacity = gbuffer["opacity"][0]
    valid = (opacity >= opacity_thr) & (depth > 0)
    # Division only where opacity >= opacity_thr > 0 (no epsilon).
    z = depth / torch.where(valid, opacity, torch.ones_like(opacity))

    with torch.no_grad():
        if valid.any():
            fallback = z[valid].median()
        else:
            # Nothing trusted (e.g. a nearly empty map): median over every
            # pixel that has some coverage. Exact zeros are "no Gaussian".
            covered = (opacity > 0) & (depth > 0)
            if not covered.any():
                return None, valid
            fallback = (depth[covered] / opacity[covered]).median()
    z = torch.where(valid, z, fallback)
    h, w = depth.shape
    rays = pixel_rays(camera, h, w, depth.device, depth.dtype)
    return rays * z[None], valid
