# Albedo of the Gaussians created for a new keyframe (docs/DECISIONS.md D7).
#
# Called by BackEnd.add_next_kf only when a shader is active; original MonoGS
# (shader None) keeps its own color initialization untouched.
#
# Every strategy has the SAME signature, rich enough for all three planned
# strategies so adding median_depth / rendered_depth does not change callers:
#
#   strategy(params, config, viewpoint, placement_depth, *, shader, render_keyframe)
#     params          - Light.init_albedo block minus `strategy` (validated by
#                       the strategy against its declared keys).
#     config          - full config (LightTracking: exposure, color space).
#     viewpoint       - the keyframe Camera: observed image, intrinsics, pose,
#                       tracked exposure_a/b.
#     placement_depth - (H,W) numpy depth used to place the new Gaussians
#                       (FrontEnd.add_new_keyframe): sensor depth, or for
#                       monocular a GUESS (flat plane, or rendered depth with
#                       outliers replaced and noise added). 0 = no Gaussian.
#     shader          - the active shader, to evaluate shading at a depth.
#     render_keyframe - zero-argument callable: render_pkg of the map at this
#                       keyframe BEFORE the new Gaussians are added, or None
#                       if the map is empty (first keyframe). Lazy, so
#                       strategies that do not need it cost no extra render.
#   returns (3,H,W) float tensor: LINEAR albedo per pixel, unclamped (values
#   above 1 are kept; see D1). GaussianModel samples it at each new
#   Gaussian's pixel.
#
# Why several strategies: for monocular the placement depth is a guess, and
# dividing the observation by the shading at a guessed depth makes the
# albedo wrong by the square of the depth ratio (inverse-square falloff).

import torch

from light_models import read_albedo_init, read_light_tracking
from utils.color_space import sRGB2Linear
from utils.slam_utils import get_observed_in_loss_space, undo_exposure_affine


def observed_radiance_linear(config, viewpoint):
    # The observation brought back to the map's exposure-neutral reference
    # (inverse affine, in the loss space where the affine is defined, D3),
    # then expressed in linear space.
    observed = get_observed_in_loss_space(config, viewpoint)
    observed = undo_exposure_affine(config, observed, viewpoint)
    if read_light_tracking(config).loss_color_space == "srgb":
        observed = sRGB2Linear(observed)
    return observed


def _observed(params, config, viewpoint, placement_depth, *, shader, render_keyframe):
    # Shading taken as 1: the albedo starts as the observed linear radiance.
    # Biased (bright near the lamp axis, dark at the cone edge) but needs no
    # depth at all; mapping then corrects it through the shaded loss.
    return observed_radiance_linear(config, viewpoint)


# name -> (function, allowed parameter keys). A name listed in
# light_models.ALBEDO_INIT_STRATEGIES but missing here is planned, not
# implemented yet.
_REGISTRY = {
    "observed": (_observed, frozenset()),
}


def get_albedo_init(config):
    """Resolves Light.init_albedo to a callable with the strategy signature
    (params already bound). Raises for unknown, unimplemented or
    mis-parameterized strategies; slam.py calls it at startup to fail fast."""
    settings = read_albedo_init(config)
    if settings.strategy not in _REGISTRY:
        raise NotImplementedError(
            f"Light.init_albedo.strategy '{settings.strategy}' is planned but "
            "not implemented yet"
        )
    fn, allowed = _REGISTRY[settings.strategy]
    unknown = set(settings.params) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in Light.init_albedo for strategy "
            f"'{settings.strategy}' (allowed: {sorted(allowed)})"
        )

    def init_albedo(config, viewpoint, placement_depth, *, shader, render_keyframe):
        with torch.no_grad():
            albedo = fn(
                settings.params,
                config,
                viewpoint,
                placement_depth,
                shader=shader,
                render_keyframe=render_keyframe,
            )
        _, h, w = viewpoint.original_image.shape
        if tuple(albedo.shape) != (3, h, w):
            raise ValueError(
                f"init_albedo '{settings.strategy}' returned shape "
                f"{tuple(albedo.shape)}, expected {(3, h, w)}"
            )
        return albedo

    return init_albedo
