# NormalSource: per-pixel surface normals in the camera frame, (3,H,W), plus
# a validity mask (H,W). Normals face the camera (n . x_C < 0).

import torch

from light_models.registry import declare_planned, register


@register("normals", "depth_fd")
class DepthFiniteDifference:
    """Normals from central differences of the back-projected points, i.e.
    of the rasterized depth AFTER division by opacity (gbuffer.py, D5/D15).
    Same construction as Replica/scripts/physics_check/common.py
    (depth_normals): n = (P[u+1]-P[u-1]) x (P[v+1]-P[v-1]), flipped to face
    the camera. Invalid if the pixel or one of its 4 neighbours is invalid,
    on the image border, or if the cross product is exactly zero."""

    def __call__(self, points, valid, gbuffer):
        _, h, w = points.shape
        du = torch.zeros_like(points)
        dv = torch.zeros_like(points)
        du[:, :, 1:-1] = points[:, :, 2:] - points[:, :, :-2]
        dv[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]
        ok = valid.clone()
        ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
        ok[1:-1, :] &= valid[2:, :] & valid[:-2, :]
        ok[[0, -1], :] = False
        ok[:, [0, -1]] = False

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

    def __call__(self, points, valid, gbuffer):
        if "normal" not in gbuffer or "normal_valid" not in gbuffer:
            raise KeyError(
                "normal_source 'gbuffer' needs gbuffer['normal'] and "
                "gbuffer['normal_valid'] (only the verification scripts supply them)"
            )
        return gbuffer["normal"], gbuffer["normal_valid"] & valid


# shortest_axis: normal = shortest axis of each Gaussian, rasterized.
declare_planned("normals", "shortest_axis")
