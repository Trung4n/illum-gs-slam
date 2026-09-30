"""light_models.build_shader: strict reading of the Light block (no torch needed)."""
import glob
import os
import pickle

import pytest
import yaml

from light_models import build_shader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config(light=None, spherical_harmonics=False):
    cfg = {"Training": {"spherical_harmonics": spherical_harmonics}}
    if light is not None:
        cfg["Light"] = light
    return cfg


def test_missing_light_block_raises():
    with pytest.raises(KeyError, match="Light"):
        build_shader(_config())


def test_missing_enabled_key_raises():
    with pytest.raises(KeyError, match="Light.enabled"):
        build_shader(_config(light={}))


@pytest.mark.parametrize("value", ["false", 0, None])
def test_enabled_must_be_bool(value):
    with pytest.raises(TypeError):
        build_shader(_config(light={"enabled": value}))


def test_disabled_returns_none():
    assert build_shader(_config(light={"enabled": False})) is None


def test_enabled_rejects_spherical_harmonics():
    with pytest.raises(ValueError, match="spherical_harmonics"):
        build_shader(
            _config(
                light={"enabled": True, "shader": {"type": "identity"}},
                spherical_harmonics=True,
            )
        )


def test_enabled_requires_shader_block():
    with pytest.raises(KeyError, match="Light.shader"):
        build_shader(_config(light={"enabled": True}))


def test_enabled_requires_shader_type():
    with pytest.raises(KeyError, match="Light.shader.type"):
        build_shader(_config(light={"enabled": True, "shader": {}}))


def test_identity_shader_returns_albedo_unchanged():
    shader = build_shader(_config(light={"enabled": True, "shader": {"type": "identity"}}))
    albedo = object()
    out = shader({"albedo": albedo, "depth": None, "opacity": None}, None)
    assert out == {"radiance_linear": albedo}


def test_shader_survives_pickling():
    # Backend and GUI processes are started with "spawn".
    shader = build_shader(_config(light={"enabled": True, "shader": {"type": "identity"}}))
    clone = pickle.loads(pickle.dumps(shader))
    assert clone({"albedo": 1}, None) == {"radiance_linear": 1}


def test_colocated_requires_its_components():
    # colocated reads intensity/pose/components from the Light block.
    with pytest.raises(KeyError, match="Light.intensity"):
        build_shader(
            _config(light={"enabled": True, "shader": {"type": "colocated"}}),
            params_data={},
        )


@pytest.mark.parametrize("name", ["Identity", "lambert", None])
def test_unknown_shader_type_raises(name):
    with pytest.raises(ValueError, match="Light.shader.type"):
        build_shader(_config(light={"enabled": True, "shader": {"type": name}}))


def test_identity_rejects_parameters():
    with pytest.raises(ValueError, match="Unknown keys"):
        build_shader(
            _config(light={"enabled": True, "shader": {"type": "identity", "gain": 2}})
        )


def test_reparam_config_resolves(monkeypatch):
    # configs/light/reparam_identity.yaml, resolved exactly as slam.py does
    # (inherit_from paths are relative to the repo root).
    from light_models import read_albedo_init, read_light_tracking
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config("configs/light/reparam_identity.yaml")
    assert type(build_shader(cfg)).__name__ == "IdentityShader"
    assert read_albedo_init(cfg).strategy == "observed"
    lt = read_light_tracking(cfg)
    assert (lt.loss_color_space, lt.exposure_affine) == ("srgb", True)
    # Everything else must be the baseline's.
    base = load_config("configs/mono/replica/office0.yaml")
    for key in ("Light", "LightTracking", "Verify", "inherit_from"):
        cfg.pop(key, None)
        base.pop(key, None)
    assert cfg == base


@pytest.mark.parametrize(
    "path",
    sorted(
        glob.glob(os.path.join(REPO, "configs", "**", "base_config.yaml"), recursive=True)
        + glob.glob(os.path.join(REPO, "configs", "live", "*.yaml"))
    ),
)
def test_every_root_config_states_baseline_explicitly(path):
    with open(path, "r") as f:
        cfg = yaml.full_load(f)
    assert cfg["Light"]["enabled"] is False, path
    # Background moved from slam.py into the config (docs/DECISIONS.md D9):
    # the baseline value must stay exactly the old hard-coded one.
    assert cfg["pipeline_params"]["background"] == [0, 0, 0], path
    # Loss options that reproduce the baseline (docs/DECISIONS.md D2, D3).
    assert cfg["LightTracking"] == {
        "loss_color_space": "srgb",
        "exposure_affine": True,
        "pixel_weight": {"enabled": False},
        "saturation_mask": {"enabled": False},
        "min_gradient": {"enabled": False},
    }, path


@pytest.mark.parametrize(
    "name, light_enabled",
    [
        ("configs/light/baseline_satmask.yaml", False),
        ("configs/light/4_2_nocos_sat.yaml", True),
        ("configs/light/4_2_lambert_sat.yaml", True),
    ],
)
def test_saturation_configs_resolve(monkeypatch, name, light_enabled):
    from light_models import read_light_tracking
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config(name)
    assert cfg["Light"]["enabled"] is light_enabled
    sm = read_light_tracking(cfg).saturation_mask
    assert (sm.enabled, sm.mode, sm.apply_to) == (True, "pixel", ("tracking", "mapping"))
    # The three runs must use the same threshold to be comparable.
    base = read_light_tracking(load_config("configs/light/baseline_satmask.yaml"))
    assert sm.threshold_8bit == base.saturation_mask.threshold_8bit


@pytest.mark.parametrize(
    "name, light_enabled",
    [
        ("configs/light/4_3_nocos_sat.yaml", True),
        ("configs/light/4_3_lambert_sat.yaml", True),
        ("configs/light/baseline_satmask_noaffine.yaml", False),
    ],
)
def test_step_4_3_configs(monkeypatch, name, light_enabled):
    from light_models import read_light_tracking
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config(name)
    lt = read_light_tracking(cfg)
    assert cfg["Light"]["enabled"] is light_enabled
    assert lt.exposure_affine is False and lt.saturation_mask.enabled is True


def test_min_gradient_configs(monkeypatch):
    from light_models import read_min_gradient
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    base = read_min_gradient(load_config("configs/light/baseline_satmask_mingrad.yaml"))
    light = read_min_gradient(load_config("configs/light/4_3_nocos_sat_mingrad.yaml"))
    assert base.enabled and light.enabled
    # The two runs must use the same floor to be comparable.
    assert base.threshold_8bit == light.threshold_8bit
    assert load_config("configs/light/4_3_nocos_sat_mingrad.yaml")["Light"]["enabled"] is True


@pytest.mark.parametrize("k", [2, 4, 8])
def test_lambert_stencil_configs(monkeypatch, k):
    from light_models.params import load_params_file  # noqa: F401
    from utils.config_utils import load_config

    monkeypatch.chdir(REPO)
    cfg = load_config(f"configs/light/4_3_lambert_sat_k{k}.yaml")
    assert cfg["Light"]["cosine"] == {"type": "lambert",
                                      "normal_source": {"type": "depth_fd", "stencil_px": k}}
    assert cfg["LightTracking"]["exposure_affine"] is False
