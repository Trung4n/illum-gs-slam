"""utils/slam_utils.get_light_pixel_weight (CPU, needs torch)."""
import pytest

torch = pytest.importorskip("torch")

from utils.slam_utils import get_light_pixel_weight  # noqa: E402


def _config(pixel_weight, light_enabled=True):
    return {
        "Light": {"enabled": light_enabled},
        "LightTracking": {
            "loss_color_space": "srgb",
            "exposure_affine": True,
            "pixel_weight": pixel_weight,
            "saturation_mask": {"enabled": False},
        },
    }


def _pkg():
    cos = torch.tensor([[[0.9, 0.2], [0.5, 0.31]]])
    valid = torch.tensor([[[True, True], [False, True]]])
    return {"light_cos_nl": cos, "light_normal_valid": valid}


def test_disabled_gives_no_weight():
    assert get_light_pixel_weight(_config({"enabled": False}), _pkg(), "tracking") is None


def test_weight_keeps_valid_normals_above_min_cos():
    cfg = _config({"enabled": True, "apply_to": ["tracking", "mapping"], "min_cos_nl": 0.3})
    w = get_light_pixel_weight(cfg, _pkg(), "mapping")
    torch.testing.assert_close(w, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))


def test_only_listed_losses_are_weighted():
    cfg = _config({"enabled": True, "apply_to": ["tracking"], "min_cos_nl": 0.3})
    assert get_light_pixel_weight(cfg, _pkg(), "mapping") is None
    assert get_light_pixel_weight(cfg, _pkg(), "tracking") is not None


def test_needs_lambert_output():
    cfg = _config({"enabled": True, "apply_to": ["tracking"], "min_cos_nl": 0.3})
    with pytest.raises(KeyError, match="lambert"):
        get_light_pixel_weight(cfg, {}, "tracking")


def test_no_weight_during_lambert_warmup():
    cfg = _config({"enabled": True, "apply_to": ["tracking"], "min_cos_nl": 0.3})
    pkg = {"light_cosine_warmup": torch.ones(1, 2, 2, dtype=torch.bool)}
    assert get_light_pixel_weight(cfg, pkg, "tracking") is None
