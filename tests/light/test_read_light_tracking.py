"""light_models.read_light_tracking: strict reading of the LightTracking block."""
import pytest

from light_models import read_light_tracking


def _config(light_tracking=None, light_enabled=False):
    cfg = {"Light": {"enabled": light_enabled}, "Training": {"spherical_harmonics": False}}
    if light_tracking is not None:
        cfg["LightTracking"] = light_tracking
    return cfg


def test_missing_block_raises():
    with pytest.raises(KeyError, match="LightTracking"):
        read_light_tracking(_config())


@pytest.mark.parametrize("missing", ["loss_color_space", "exposure_affine"])
def test_missing_key_raises(missing):
    block = {"loss_color_space": "srgb", "exposure_affine": True}
    del block[missing]
    with pytest.raises(KeyError, match=f"LightTracking.{missing}"):
        read_light_tracking(_config(block))


@pytest.mark.parametrize("space", ["sRGB", "lin", "", None])
def test_unknown_color_space_raises(space):
    with pytest.raises(ValueError, match="loss_color_space"):
        read_light_tracking(_config({"loss_color_space": space, "exposure_affine": True}))


@pytest.mark.parametrize("value", ["true", 1, None])
def test_exposure_affine_must_be_bool(value):
    with pytest.raises(TypeError, match="exposure_affine"):
        read_light_tracking(_config({"loss_color_space": "srgb", "exposure_affine": value}))


def test_linear_requires_light_enabled():
    with pytest.raises(ValueError, match="Light.enabled"):
        read_light_tracking(_config({"loss_color_space": "linear", "exposure_affine": False}))


def test_linear_needs_light_block_too():
    cfg = {"LightTracking": {"loss_color_space": "linear", "exposure_affine": False}}
    with pytest.raises(KeyError, match="Light"):
        read_light_tracking(cfg)


@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize("affine", [True, False])
def test_valid_combinations(space, affine):
    s = read_light_tracking(
        _config({"loss_color_space": space, "exposure_affine": affine}, light_enabled=True)
    )
    assert s.loss_color_space == space
    assert s.exposure_affine is affine


def test_baseline_values_accepted_with_light_disabled():
    s = read_light_tracking(_config({"loss_color_space": "srgb", "exposure_affine": True}))
    assert (s.loss_color_space, s.exposure_affine) == ("srgb", True)
