# Entry point of the co-located light model (CLAUDE.md section 4).
#
# build_shader(config) is the ONLY place that decides whether SLAM runs the
# original MonoGS image formation or the lit one. Its result is handed to
# every render() call as the required `shader` keyword (see
# gaussian_renderer.render for the shader contract).
#
# read_light_tracking(config) parses the LightTracking block that the loss
# functions in utils/slam_utils.py depend on.
#
# Every key is read strictly: a missing key raises, nothing falls back to a
# "reasonable" default (CLAUDE.md section 2). This module does not import
# torch, so the config parsing is unit-testable without a GPU environment.

from types import SimpleNamespace

from light_models.shaders import make_shader
from utils.config_utils import require_key

# Spaces the photometric loss can be computed in (docs/DECISIONS.md D2):
#   srgb   - render_pkg["render"] vs the observed image as stored.
#   linear - render_pkg["radiance_linear"] vs sRGB2Linear(observed).
LOSS_COLOR_SPACES = ("srgb", "linear")


def _light_enabled(config):
    enabled = require_key(require_key(config, "Light", ""), "enabled", "Light")
    if not isinstance(enabled, bool):
        raise TypeError(f"Light.enabled must be true or false, got {enabled!r}")
    return enabled


def build_shader(config):
    """Returns None for original MonoGS (Light.enabled: false), otherwise the
    shader callable used by render()."""
    if not _light_enabled(config):
        return None

    # Albedo is stored in the degree-0 SH coefficient (docs/DECISIONS.md):
    # higher SH bands would let the map bake view-dependent lighting back in.
    if require_key(config["Training"], "spherical_harmonics", "Training"):
        raise ValueError(
            "Light.enabled: true requires Training.spherical_harmonics: false"
        )

    return make_shader(require_key(config["Light"], "shader", "Light"))


# Strategies for the albedo of newly created Gaussians (docs/DECISIONS.md D7).
# The implementations live in light_models/albedo_init.py (needs torch); only
# the names are here so the config parsing stays importable without torch.
#   observed       - albedo = linearized observation, shading taken as 1.
#   median_depth   - observation / shading at the keyframe's median depth.
#   rendered_depth - observation / shading at the depth rendered from the map.
ALBEDO_INIT_STRATEGIES = ("observed", "median_depth", "rendered_depth")


def read_albedo_init(config):
    """Light.init_albedo block: {strategy: <name>, <strategy-specific keys>}.

    Returns a namespace with `strategy` and `params` (every other key of the
    block, validated by the strategy itself in albedo_init.py).
    """
    light_cfg = require_key(config, "Light", "")
    block = require_key(light_cfg, "init_albedo", "Light")
    strategy = require_key(block, "strategy", "Light.init_albedo")
    if strategy not in ALBEDO_INIT_STRATEGIES:
        raise ValueError(
            f"Light.init_albedo.strategy must be one of {ALBEDO_INIT_STRATEGIES}, "
            f"got {strategy!r}"
        )
    params = {k: v for k, v in block.items() if k != "strategy"}
    return SimpleNamespace(strategy=strategy, params=params)


def read_light_tracking(config):
    """LightTracking options used by the tracking AND mapping losses.

    Returns a namespace with:
      loss_color_space: "srgb" | "linear" (see LOSS_COLOR_SPACES).
      exposure_affine: bool. If true, exp(a) * I + b is applied to the
        prediction in the loss color space right before the loss
        (docs/DECISIONS.md D3); if false, no affine term at all.
    """
    block = require_key(config, "LightTracking", "")
    space = require_key(block, "loss_color_space", "LightTracking")
    if space not in LOSS_COLOR_SPACES:
        raise ValueError(
            f"LightTracking.loss_color_space must be one of {LOSS_COLOR_SPACES}, "
            f"got {space!r}"
        )
    affine = require_key(block, "exposure_affine", "LightTracking")
    if not isinstance(affine, bool):
        raise TypeError(
            f"LightTracking.exposure_affine must be true or false, got {affine!r}"
        )
    if space == "linear" and not _light_enabled(config):
        # radiance_linear only exists when a shader runs. Linearizing the
        # baseline's sRGB render instead would be a different model, not a
        # silent fallback we want.
        raise ValueError(
            "LightTracking.loss_color_space: linear requires Light.enabled: true"
        )
    return SimpleNamespace(loss_color_space=space, exposure_affine=affine)
