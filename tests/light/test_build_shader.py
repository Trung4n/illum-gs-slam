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
    assert shader({"albedo": albedo, "depth": None, "opacity": None}, None) is albedo


def test_shader_survives_pickling():
    # Backend and GUI processes are started with "spawn".
    shader = build_shader(_config(light={"enabled": True, "shader": {"type": "identity"}}))
    clone = pickle.loads(pickle.dumps(shader))
    assert clone({"albedo": 1}, None) == 1


def test_planned_shader_not_implemented():
    with pytest.raises(NotImplementedError, match="colocated"):
        build_shader(_config(light={"enabled": True, "shader": {"type": "colocated"}}))


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
    }, path
