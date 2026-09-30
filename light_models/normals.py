# NormalSource: per-pixel surface normals in the camera frame, (3,H,W), plus
# a validity mask (H,W). Normals face the camera (n . x_C < 0).

import torch

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

    def __init__(self, stencil_px):
        if isinstance(stencil_px, bool) or not isinstance(stencil_px, int) or stencil_px < 1:
            raise ValueError(
                f"Light.cosine.normal_source.stencil_px must be an integer >= 1, got {stencil_px!r}"
            )
        self.k = stencil_px

    def __call__(self, points, valid, gbuffer):
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

    def __call__(self, points, valid, gbuffer):
        if "normal" not in gbuffer or "normal_valid" not in gbuffer:
            raise KeyError(
                "normal_source 'gbuffer' needs gbuffer['normal'] and "
                "gbuffer['normal_valid'] (only the verification scripts supply them)"
            )
        return gbuffer["normal"], gbuffer["normal_valid"] & valid


# shortest_axis: normal = shortest axis of each Gaussian, rasterized.
declare_planned("normals", "shortest_axis")
