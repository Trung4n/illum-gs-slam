# The co-located light shader (CLAUDE.md section 4.1): a lamp rigidly
# mounted on the camera, all lighting geometry in the camera frame.
#
#   x_C      = surface point from the G-buffer (gbuffer.py)
#   v        = x_C - t_CL,  d_l = |v|,  l = -v / d_l      (point -> light)
#   cos th   = (v / d_l) . a_L,  a_L = R_CL e_z           (light axis)
#   direct   = Intensity * Angular(cos th) * Falloff(d_l) * Cosine(n, l)
#   I_hat    = Ambient.combine(A, direct)
#
# Every form and number comes from the Light config and the per-scene params
# file (ParamSpec); nothing about the lamp is written here. The pose
# dependence is only through the rasterized maps, so gradients reach the
# camera pose through the rasterizer as before; the light parameters are
# never optimized in tracking (they are fixed CPU tensors here).

import torch

# Imported for their @register side effect: the component forms must be in
# the registry before build_component looks them up.
import light_models.ambient  # noqa: F401
import light_models.angular  # noqa: F401
import light_models.cosine  # noqa: F401
import light_models.falloff  # noqa: F401
import light_models.normals  # noqa: F401
from light_models.gbuffer import surface_points
from light_models.params import resolve_param
from light_models.registry import build_component
from utils.config_utils import require_key


def _vector(spec, shape):
    t = spec.tensor()
    if tuple(t.shape) != shape:
        raise ValueError(f"{spec.name} must have shape {shape}, got {tuple(t.shape)}")
    return t


class ColocatedShader:
    def __init__(self, light_cfg, params_data):
        where = "Light"
        intensity = resolve_param(
            require_key(light_cfg, "intensity", where), "Light.intensity", params_data
        )
        k = intensity.tensor()
        if k.dim() == 1 and k.shape[0] == 3:
            k = k.view(3, 1, 1)
        elif k.dim() != 0:
            raise ValueError(f"Light.intensity must be a number or 3 numbers, got {intensity.value!r}")
        self.intensity = k

        pose = require_key(light_cfg, "pose", where)
        self.t_CL = _vector(
            resolve_param(require_key(pose, "t_CL", "Light.pose"), "Light.pose.t_CL", params_data),
            (3,),
        )
        r_cl = _vector(
            resolve_param(require_key(pose, "R_CL", "Light.pose"), "Light.pose.R_CL", params_data),
            (3, 3),
        )
        # a_L = R_CL e_z, normalized (R_CL from a fit is only ~orthonormal).
        axis = r_cl[:, 2]
        self.axis = axis / axis.norm()

        self.falloff = build_component(
            "falloff", require_key(light_cfg, "falloff", where), "Light.falloff", params_data
        )
        self.angular = build_component(
            "angular", require_key(light_cfg, "angular", where), "Light.angular", params_data
        )
        self.cosine = build_component(
            "cosine", require_key(light_cfg, "cosine", where), "Light.cosine", params_data
        )
        self.ambient = build_component(
            "ambient", require_key(light_cfg, "ambient", where), "Light.ambient", params_data
        )

        gb = require_key(light_cfg, "gbuffer", where)
        thr = require_key(gb, "opacity_thr", "Light.gbuffer")
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not 0 < thr <= 1:
            raise ValueError(f"Light.gbuffer.opacity_thr must be in (0, 1], got {thr!r}")
        extra = set(gb) - {"opacity_thr"}
        if extra:
            raise ValueError(f"Light.gbuffer: unexpected keys {sorted(extra)}")
        self.opacity_thr = float(thr)

    def __call__(self, gbuffer, viewpoint_camera):
        albedo = gbuffer["albedo"]
        points, valid = surface_points(gbuffer, viewpoint_camera, self.opacity_thr)
        if points is None:
            # Empty map: nothing to shade (albedo is all zero there anyway).
            zeros = torch.zeros_like(valid, dtype=albedo.dtype)
            return {"radiance_linear": albedo * 0.0, "light_valid": valid[None],
                    "light_direct": zeros[None]}
        dev, dt = points.device, points.dtype
        t = self.t_CL.to(dev, dt).view(3, 1, 1)
        axis = self.axis.to(dev, dt).view(3, 1, 1)

        v = points - t
        dist = v.norm(dim=0)  # > 0: points lie in front of the camera
        to_light = -v / dist[None]
        cos_theta = (v * axis).sum(0) / dist

        phi = self.angular(cos_theta)
        fall = self.falloff(dist)
        cos_term, aux = self.cosine(points, valid, to_light, gbuffer)
        k = self.intensity.to(dev, dt)
        direct = k * (phi * fall * cos_term)
        radiance = self.ambient(albedo, direct)

        out = {
            "radiance_linear": radiance,
            "light_valid": valid[None],
            "light_direct": (direct if direct.dim() == 3 else direct[None]),
        }
        for name, value in aux.items():
            out["light_" + name] = value[None]
        return out
