"""light_models/albedo_init.py (needs torch; the image tests also need CUDA).
The pixel labelling of new Gaussians is tested in tests/test_pixel_index.py.

    python -m pytest tests/test_albedo_init.py
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from light_models.albedo_init import get_albedo_init  # noqa: E402
from utils.color_space import linear2sRGB, sRGB2Linear  # noqa: E402

H, W = 6, 9
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _config(strategy="observed", space="srgb", affine=False, **params):
    return {
        "Light": {"enabled": True, "init_albedo": {"strategy": strategy, **params}},
        "LightTracking": {
            "loss_color_space": space,
            "exposure_affine": affine,
            "pixel_weight": {"enabled": False},
            "saturation_mask": {"enabled": False},
        },
    }


def _viewpoint():
    g = torch.Generator().manual_seed(0)
    return SimpleNamespace(
        original_image=torch.rand(3, H, W, generator=g),
        exposure_a=torch.tensor([0.2], device="cuda"),
        exposure_b=torch.tensor([0.03], device="cuda"),
    )


def _call(config, vp, render_keyframe=None):
    def no_render():
        raise AssertionError("observed must not render the map")

    return get_albedo_init(config)(
        config,
        vp,
        np.ones((H, W), np.float32),
        shader=object(),
        render_keyframe=render_keyframe or no_render,
    )


@pytest.mark.parametrize("name", ["median_depth", "rendered_depth", "placement_depth"])
def test_deshading_strategies_require_min_shading(name):
    with pytest.raises(KeyError, match="min_shading"):
        get_albedo_init(_config(name))


@pytest.mark.parametrize("value", [0, -0.1, "0.05", True])
def test_min_shading_must_be_positive(value):
    with pytest.raises(ValueError, match="min_shading"):
        get_albedo_init(_config("median_depth", min_shading=value))


def _colocated_shader():
    from light_models import build_shader

    p = lambda key: {"source": "params_file", "key": key, "learnable": False}  # noqa: E731
    params = {"K": 30.0, "t": [0.0, -0.05, 0.0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              "a": 30.0, "b": 0.35, "c": 0.12}
    cfg = {
        "Training": {"spherical_harmonics": False},
        "Light": {
            "enabled": True, "shader": {"type": "colocated"}, "intensity": p("K"),
            "pose": {"t_CL": p("t"), "R_CL": p("R")}, "falloff": {"type": "inverse_square"},
            "angular": {"type": "blender_spot", "half_angle_deg": p("a"), "blend": p("b")},
            "cosine": {"type": "none"},
            "ambient": {"type": "multiplicative_const", "c": p("c")},
            "gbuffer": {"opacity_thr": 0.5},
        },
    }
    return build_shader(cfg, params)


def test_deshading_inverts_the_shader(monkeypatch):
    # observation = albedo * shading(z)  =>  _deshade gives the albedo back.
    import light_models.albedo_init as ai

    shader = _colocated_shader()
    cam = SimpleNamespace(fx=40.0, fy=40.0, cx=4.5, cy=2.5, original_image=torch.zeros(3, H, W))
    z = torch.full((H, W), 1.5, dtype=torch.float64)
    albedo = torch.rand(3, H, W, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    observed = albedo * ai.shading_at_depth(shader, z, cam)
    monkeypatch.setattr(ai, "observed_radiance_linear", lambda config, vp: observed)
    got = ai._deshade({"min_shading": 0.05}, None, cam, z, shader)
    torch.testing.assert_close(got, albedo)


def test_median_depth_uses_one_plane(monkeypatch):
    import light_models.albedo_init as ai

    shader = _colocated_shader()
    cam = SimpleNamespace(fx=40.0, fy=40.0, cx=4.5, cy=2.5, original_image=torch.zeros(3, H, W))
    placement = np.random.default_rng(0).uniform(1.0, 3.0, (H, W)).astype(np.float32)
    placement[0, 0] = 0.0  # no Gaussian there: ignored by the median
    monkeypatch.setattr(ai, "observed_radiance_linear",
                        lambda config, vp: torch.ones(3, H, W))
    got = ai._median_depth({"min_shading": 0.05}, None, cam, placement,
                           shader=shader, render_keyframe=None)
    plane = torch.full((H, W), float(np.median(placement[placement > 0])))
    torch.testing.assert_close(got, 1.0 / ai.shading_at_depth(shader, plane, cam))


def test_observed_rejects_unknown_params():
    with pytest.raises(ValueError, match="Unknown keys"):
        get_albedo_init(_config("observed", opacity_thr=0.5))


@needs_cuda
def test_observed_without_affine_is_linearized_observation():
    vp = _viewpoint()
    got = _call(_config(affine=False), vp)
    torch.testing.assert_close(got, sRGB2Linear(vp.original_image.cuda()))


@needs_cuda
def test_observed_undoes_affine_in_srgb_loss_space():
    # The albedo, rendered with unit shading and pushed through the same
    # exposure affine as the loss, must give back the observation.
    vp = _viewpoint()
    albedo = _call(_config(space="srgb", affine=True), vp)
    reobserved = torch.exp(vp.exposure_a) * linear2sRGB(albedo) + vp.exposure_b
    torch.testing.assert_close(reobserved, vp.original_image.cuda())


@needs_cuda
def test_observed_undoes_affine_in_linear_loss_space():
    vp = _viewpoint()
    albedo = _call(_config(space="linear", affine=True), vp)
    reobserved = torch.exp(vp.exposure_a) * albedo + vp.exposure_b
    torch.testing.assert_close(reobserved, sRGB2Linear(vp.original_image.cuda()))


def test_rendered_depth_on_empty_map_ignores_placement_noise(monkeypatch):
    # First keyframe: no rendered depth. The noisy placement depth must not
    # be de-shaded pixel by pixel (its finite-difference normals are
    # meaningless, D33): same result as the median plane.
    import light_models.albedo_init as ai

    shader = _colocated_shader()
    cam = SimpleNamespace(fx=40.0, fy=40.0, cx=4.5, cy=2.5, original_image=torch.zeros(3, H, W))
    rng = np.random.default_rng(1)
    placement = (2.0 + rng.normal(0.0, 0.3, (H, W))).astype(np.float32)
    monkeypatch.setattr(ai, "observed_radiance_linear", lambda config, vp: torch.ones(3, H, W))
    params = {"min_shading": 0.05}
    got = ai._rendered_depth(params, None, cam, placement, shader=shader, render_keyframe=lambda: None)
    expected = ai._median_depth(params, None, cam, placement, shader=shader, render_keyframe=None)
    torch.testing.assert_close(got, expected)


def test_placement_depth_deshades_each_pixel_at_its_depth(monkeypatch):
    # RGB-D (D47): observation = albedo * shading(sensor depth), pixel by
    # pixel, so de-shading at the placement depth gives the albedo back.
    import light_models.albedo_init as ai

    shader = _colocated_shader()
    cam = SimpleNamespace(fx=40.0, fy=40.0, cx=4.5, cy=2.5, original_image=torch.zeros(3, H, W))
    placement = np.random.default_rng(2).uniform(1.0, 3.0, (H, W)).astype(np.float32)
    z = torch.from_numpy(placement)
    albedo = torch.rand(3, H, W, generator=torch.Generator().manual_seed(3))
    observed = albedo * ai.shading_at_depth(shader, z, cam)
    monkeypatch.setattr(ai, "observed_radiance_linear", lambda config, vp: observed)
    got = ai._placement_depth({"min_shading": 0.05}, None, cam, placement,
                              shader=shader, render_keyframe=None)
    torch.testing.assert_close(got, albedo)


@pytest.mark.parametrize("sensor, ok", [("depth", True), ("monocular", False)])
def test_placement_depth_refused_for_monocular(sensor, ok):
    cfg = _config("placement_depth", min_shading=0.05)
    cfg["Dataset"] = {"sensor_type": sensor}
    if ok:
        get_albedo_init(cfg)
    else:
        with pytest.raises(ValueError, match="measured"):
            get_albedo_init(cfg)
