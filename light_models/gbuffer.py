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


def pixel_rays(camera, height, width, device, dtype, pixel_offset):
    """(3,H,W) K^-1 [u + o, v + o, 1] for pixel (row v, column u), K in the
    OpenCV convention of the data (integer pixel centers). o = pixel_offset
    is where the map's pixel i sits on that grid: 0 for maps in the data's
    own convention (ground-truth passes, the placement depth), 0.5 for
    rasterized maps (RASTERIZER_PIXEL_OFFSET, docs/DECISIONS.md D29)."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = (u + pixel_offset - camera.cx) / camera.fx
    y = (v + pixel_offset - camera.cy) / camera.fy
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
    # Required: which pixel convention the maps follow (D29).
    offset = gbuffer["pixel_offset"]
    rays = pixel_rays(camera, h, w, depth.device, depth.dtype, offset)
    return rays * z[None], valid
