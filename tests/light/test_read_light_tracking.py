"""light_models.read_light_tracking: strict reading of the LightTracking block."""
import pytest

from light_models import read_light_tracking


def _config(light_tracking=None, light_enabled=False):
    cfg = {"Light": {"enabled": light_enabled}, "Training": {"spherical_harmonics": False}}
    if light_tracking is not None:
        # Tests about other keys get the baseline pixel_weight block.
        cfg["LightTracking"] = {
            "pixel_weight": {"enabled": False},
            "saturation_mask": {"enabled": False},
            **light_tracking,
        }
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


def _lt(pixel_weight, saturation_mask=None):
    return {
        "loss_color_space": "srgb",
        "exposure_affine": False,
        "pixel_weight": pixel_weight,
        "saturation_mask": saturation_mask or {"enabled": False},
    }


def test_pixel_weight_block_required():
    cfg = {"Light": {"enabled": True},
           "LightTracking": {"loss_color_space": "srgb", "exposure_affine": False,
                             "saturation_mask": {"enabled": False}}}
    with pytest.raises(KeyError, match="LightTracking.pixel_weight"):
        read_light_tracking(cfg)


def test_pixel_weight_disabled():
    pw = read_light_tracking(_config(_lt({"enabled": False}))).pixel_weight
    assert pw.enabled is False and pw.apply_to == ()


def test_pixel_weight_enabled_requires_light():
    block = {"enabled": True, "apply_to": ["tracking"], "min_cos_nl": 0.3}
    with pytest.raises(ValueError, match="Light.enabled"):
        read_light_tracking(_config(_lt(block), light_enabled=False))


@pytest.mark.parametrize("missing", ["apply_to", "min_cos_nl"])
def test_pixel_weight_enabled_keys_required(missing):
    block = {"enabled": True, "apply_to": ["tracking"], "min_cos_nl": 0.3}
    del block[missing]
    with pytest.raises(KeyError, match=f"pixel_weight.{missing}"):
        read_light_tracking(_config(_lt(block), light_enabled=True))


@pytest.mark.parametrize("apply_to", [[], ["track"], ["tracking", "tracking"], "tracking"])
def test_pixel_weight_apply_to_validated(apply_to):
    block = {"enabled": True, "apply_to": apply_to, "min_cos_nl": 0.3}
    with pytest.raises(ValueError, match="apply_to"):
        read_light_tracking(_config(_lt(block), light_enabled=True))


@pytest.mark.parametrize("value", [1.5, -2, "0.3", True])
def test_pixel_weight_min_cos_validated(value):
    block = {"enabled": True, "apply_to": ["mapping"], "min_cos_nl": value}
    with pytest.raises(ValueError, match="min_cos_nl"):
        read_light_tracking(_config(_lt(block), light_enabled=True))


def test_pixel_weight_unknown_key_rejected():
    block = {"enabled": False, "saturation_thr": 250}
    with pytest.raises(ValueError, match="unexpected keys"):
        read_light_tracking(_config(_lt(block)))


def test_pixel_weight_enabled_parsed():
    block = {"enabled": True, "apply_to": ["tracking", "mapping"], "min_cos_nl": 0.3}
    pw = read_light_tracking(_config(_lt(block), light_enabled=True)).pixel_weight
    assert (pw.enabled, pw.apply_to, pw.min_cos_nl) == (True, ("tracking", "mapping"), 0.3)


_SAT = {"enabled": True, "mode": "pixel", "threshold_8bit": 250, "apply_to": ["tracking", "mapping"]}


def test_saturation_block_required():
    cfg = {"Light": {"enabled": False},
           "LightTracking": {"loss_color_space": "srgb", "exposure_affine": True,
                             "pixel_weight": {"enabled": False}}}
    with pytest.raises(KeyError, match="LightTracking.saturation_mask"):
        read_light_tracking(cfg)


def test_saturation_works_with_light_disabled():
    sm = read_light_tracking(_config(_lt({"enabled": False}, _SAT))).saturation_mask
    assert (sm.enabled, sm.mode, sm.threshold_8bit, sm.apply_to) == (
        True, "pixel", 250, ("tracking", "mapping"))


@pytest.mark.parametrize("missing", ["mode", "threshold_8bit", "apply_to"])
def test_saturation_keys_required(missing):
    block = dict(_SAT)
    del block[missing]
    with pytest.raises(KeyError, match=f"saturation_mask.{missing}"):
        read_light_tracking(_config(_lt({"enabled": False}, block)))


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"mode": "any"}, "mode"),
        ({"threshold_8bit": 250.0}, "threshold_8bit"),
        ({"threshold_8bit": 0}, "threshold_8bit"),
        ({"threshold_8bit": 256}, "threshold_8bit"),
        ({"threshold_8bit": True}, "threshold_8bit"),
        ({"apply_to": []}, "apply_to"),
        ({"apply_to": ["refinement"]}, "apply_to"),
        ({"extra": 1}, "unexpected keys"),
    ],
)
def test_saturation_values_validated(patch, match):
    with pytest.raises(ValueError, match=match):
        read_light_tracking(_config(_lt({"enabled": False}, {**_SAT, **patch})))


def test_saturation_disabled():
    sm = read_light_tracking(_config(_lt({"enabled": False}))).saturation_mask
    assert sm.enabled is False and sm.apply_to == ()
