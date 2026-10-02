# NormalSource: per-pixel surface normals in the camera frame, (3,H,W), plus
# a validity mask (H,W). Normals face the camera (n . x_C < 0).
#
# Call: source(points, valid, gbuffer, camera). requires_sensor_depth: True
# for sources that read the observed frame's depth (camera.depth); the shader
# builder refuses them for monocular (light_models.build_shader).

import torch

from light_models.gbuffer import pixel_rays
from light_models.registry import declare_planned, register


@register("normals", "depth_fd", options=("stencil_px",))
class DepthFiniteDifference:
    """Normals from central differences of the back-projected points, i.e.
    of the rasterized depth AFTER division by opacity (gbuffer.py, D5/D15):
    n = (P[u+k]-P[u-k]) x (P[v+k]-P[v-k]), k = stencil_px, flipped to face
    the camera. k = 1 is the construction of Replica/scripts/physics_check/
    common.py (depth_normals). A wider stencil averages out the bumps of a
    surface made of discrete Gaussians, which make 1-pixel differences
    unusable (layer 2 and tracking probe, docs/DECISIONS.md D40), at the cost
    of more pixels invalid near depth edges. Invalid if the pixel or one of
    its 4 neighbours at distance k is invalid, within k of the image border,
    or if the cross product is exactly zero."""

    requires_sensor_depth = False

    def __init__(self, stencil_px):
        if isinstance(stencil_px, bool) or not isinstance(stencil_px, int) or stencil_px < 1:
            raise ValueError(
                f"Light.cosine.normal_source.stencil_px must be an integer >= 1, got {stencil_px!r}"
            )
        self.k = stencil_px

    def __call__(self, points, valid, gbuffer, camera):
        k = self.k
        du = torch.zeros_like(points)
        dv = torch.zeros_like(points)
        du[:, :, k:-k] = points[:, :, 2 * k:] - points[:, :, :-2 * k]
        dv[:, k:-k, :] = points[:, 2 * k:, :] - points[:, :-2 * k, :]
        ok = valid.clone()
        ok[:, k:-k] &= valid[:, 2 * k:] & valid[:, :-2 * k]
        ok[k:-k, :] &= valid[2 * k:, :] & valid[:-2 * k, :]
        ok[:k, :] = False
        ok[-k:, :] = False
        ok[:, :k] = False
        ok[:, -k:] = False

        n = torch.cross(du, dv, dim=0)
        norm = n.norm(dim=0)
        ok &= norm > 0
        # Division only where the norm is known to be > 0 (no epsilon).
        n = n / torch.where(ok, norm, torch.ones_like(norm))
        facing_away = (n * points).sum(0) > 0
        n = torch.where(facing_away[None], -n, n)
        n = torch.where(ok[None], n, torch.zeros_like(n))
        return n, ok


@register("normals", "gbuffer")
class GBufferNormals:
    """Normals supplied in the G-buffer itself (gbuffer["normal"],
    gbuffer["normal_valid"]). SLAM does not provide them; used by the
    verification scripts to feed ground-truth normals through the SAME
    shading code (docs/DECISIONS.md D15, D18)."""

    requires_sensor_depth = False

    def __call__(self, points, valid, gbuffer, camera):
        if "normal" not in gbuffer or "normal_valid" not in gbuffer:
            raise KeyError(
                "normal_source 'gbuffer' needs gbuffer['normal'] and "
                "gbuffer['normal_valid'] (only the verification scripts supply them)"
            )
        return gbuffer["normal"], gbuffer["normal_valid"] & valid


@register("normals", "sensor_depth_fd", options=("stencil_px",))
class SensorDepthFiniteDifference:
    """depth_fd applied to the OBSERVED frame's sensor depth (camera.depth,
    RGB-D only), not to the rasterized depth (docs/DECISIONS.md D49).

    Why: the rasterizer gives every Gaussian ONE depth, that of its center,
    over its whole footprint, so the rasterized depth of a SLAM map is a
    staircase whose steps are as wide as the Gaussians; finite differences
    of it give wrong normals at every step edge (21% of the pixels with
    n . l < 0.3 in RGB-D SLAM vs 1-3% on ground truth, D48). The sensor depth
    has no steps.

    Consequences: n is a constant of the observation (no gradient), so the
    pose still reaches the shading through the rasterized depth (distance
    and direction to the lamp) and the albedo, as before. In tracking, n is
    the normal of the frame being tracked, in its own camera frame: exactly
    the n that the shading at the true pose needs. The sensor depth is in the
    data's pixel convention (offset 0) while the rasterized maps sit half a
    pixel off (D29); the normal at pixel i is used with the light direction
    of rasterized pixel i, half a pixel apart, which is negligible for a
    direction. Pixels without a sensor reading (depth 0) have no normal; the
    result is also restricted to the pixels the G-buffer trusts (`valid`)."""

    requires_sensor_depth = True

    def __init__(self, stencil_px):
        self.fd = DepthFiniteDifference(stencil_px)
        # Runtime cache, not configuration: normals of the last camera seen
        # (tracking renders the same frame for every iteration). One entry
        # only: mapping cycles through many keyframes.
        self._cache_key = None
        self._cache = None

    def __getstate__(self):
        # Shaders are pickled to the backend / GUI processes: no CUDA tensors.
        state = self.__dict__.copy()
        state["_cache_key"] = None
        state["_cache"] = None
        return state

    def __call__(self, points, valid, gbuffer, camera):
        depth = getattr(camera, "depth", None)
        if depth is None:
            raise ValueError(
                "normal_source 'sensor_depth_fd' needs the observed frame's sensor "
                "depth (camera.depth): RGB-D only"
            )
        # The key holds the camera and depth objects themselves (not their
        # id(), which Python reuses after garbage collection).
        cached = self._cache_key
        hit = (
            cached is not None
            and cached[0] is camera
            and cached[1] is depth
            and cached[2:] == (points.device, points.dtype)
        )
        if not hit:
            z = depth if torch.is_tensor(depth) else torch.from_numpy(depth)
            z = z.to(points.device, points.dtype).reshape(points.shape[1:])
            h, w = z.shape
            rays = pixel_rays(camera, h, w, points.device, points.dtype, 0.0)
            # 0 = no sensor reading (the dataset's convention), not a threshold.
            self._cache = self.fd(rays * z[None], z > 0, gbuffer, camera)
            self._cache_key = (camera, depth, points.device, points.dtype)
        n, ok = self._cache
        return n, ok & valid


# shortest_axis: normal = shortest axis of each Gaussian, rasterized.
declare_planned("normals", "shortest_axis")
