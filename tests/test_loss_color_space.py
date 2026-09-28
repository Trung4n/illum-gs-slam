"""utils/slam_utils.py losses: key selection by loss_color_space, exposure
affine in the loss space, masks on the observed image. Needs torch + CUDA
(the loss functions move tensors with .cuda()); skipped otherwise.

    python -m pytest tests/test_loss_color_space.py
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from utils.color_space import linear2sRGB, sRGB2Linear  # noqa: E402
from utils.slam_utils import (  # noqa: E402
    get_loss_images,
    get_loss_mapping,
    get_loss_tracking,
)

H, W = 8, 10
RGB_BOUNDARY = 0.01


def _config(space="srgb", affine=True, light_enabled=None):
    if light_enabled is None:
        light_enabled = space == "linear"
    return {
        "Light": {"enabled": light_enabled},
        "LightTracking": {"loss_color_space": space, "exposure_affine": affine},
        "Training": {"monocular": True, "rgb_boundary_threshold": RGB_BOUNDARY},
    }


def _viewpoint(seed=0):
    g = torch.Generator().manual_seed(seed)
    observed = torch.rand(3, H, W, generator=g)
    observed[:, 0, 0] = 0.0  # a black pixel, removed by rgb_boundary_threshold
    return SimpleNamespace(
        original_image=observed,
        grad_mask=(torch.rand(1, H, W, generator=g) > 0.3).cuda(),
        exposure_a=torch.tensor([0.1], device="cuda", requires_grad=True),
        exposure_b=torch.tensor([-0.02], device="cuda", requires_grad=True),
    )


def _render_pkg(seed=1, with_linear=False):
    g = torch.Generator().manual_seed(seed)
    pkg = {
        "depth": torch.rand(1, H, W, generator=g).cuda(),
        "opacity": torch.rand(1, H, W, generator=g).cuda(),
    }
    if with_linear:
        radiance = torch.rand(3, H, W, generator=g).cuda()
        pkg["radiance_linear"] = radiance
        pkg["render"] = linear2sRGB(radiance)
    else:
        pkg["render"] = torch.rand(3, H, W, generator=g).cuda()
    return pkg


def _baseline_tracking(pkg, vp):
    # Original MonoGS get_loss_tracking (monocular), written out.
    gt = vp.original_image.cuda()
    image = torch.exp(vp.exposure_a) * pkg["render"] + vp.exposure_b
    mask = (gt.sum(dim=0) > RGB_BOUNDARY).view(1, H, W) * vp.grad_mask
    return (pkg["opacity"] * torch.abs(image * mask - gt * mask)).mean()


def _baseline_mapping(pkg, vp):
    gt = vp.original_image.cuda()
    image = torch.exp(vp.exposure_a) * pkg["render"] + vp.exposure_b
    mask = (gt.sum(dim=0) > RGB_BOUNDARY).view(1, H, W)
    return torch.abs(image * mask - gt * mask).mean()


def test_srgb_with_affine_equals_baseline_tracking():
    pkg, vp = _render_pkg(), _viewpoint()
    got = get_loss_tracking(_config(), pkg, vp)
    assert torch.equal(got, _baseline_tracking(pkg, vp))


def test_srgb_with_affine_equals_baseline_mapping():
    pkg, vp = _render_pkg(), _viewpoint()
    got = get_loss_mapping(_config(), pkg, vp)
    assert torch.equal(got, _baseline_mapping(pkg, vp))


def test_linear_selects_radiance_and_linearized_target():
    pkg, vp = _render_pkg(with_linear=True), _viewpoint()
    pred, target = get_loss_images(_config("linear"), pkg, vp)
    assert pred is pkg["radiance_linear"]
    torch.testing.assert_close(target, sRGB2Linear(vp.original_image.cuda()))


def test_linear_without_shader_output_fails_loudly():
    # A render_pkg without "radiance_linear" must not fall back to "render".
    with pytest.raises(KeyError):
        get_loss_images(_config("linear"), _render_pkg(), _viewpoint())


@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_no_affine_means_no_exposure_gradient(space):
    pkg, vp = _render_pkg(with_linear=True), _viewpoint()
    get_loss_tracking(_config(space, affine=False), pkg, vp).backward()
    assert vp.exposure_a.grad is None and vp.exposure_b.grad is None


def test_affine_is_applied_in_the_loss_space():
    pkg, vp = _render_pkg(with_linear=True), _viewpoint()
    got = get_loss_mapping(_config("linear", affine=True), pkg, vp)
    gt = sRGB2Linear(vp.original_image.cuda())
    image = torch.exp(vp.exposure_a) * pkg["radiance_linear"] + vp.exposure_b
    mask = (vp.original_image.cuda().sum(dim=0) > RGB_BOUNDARY).view(1, H, W)
    torch.testing.assert_close(got, torch.abs(image * mask - gt * mask).mean())


def test_mapping_initialization_skips_affine():
    pkg, vp = _render_pkg(), _viewpoint()
    get_loss_mapping(_config(), pkg, vp, initialization=True).backward()
    assert vp.exposure_a.grad is None and vp.exposure_b.grad is None


def test_mask_uses_observed_image_not_linearized_target():
    # A pixel whose sRGB sum is just above the threshold falls below it once
    # linearized; it must stay in the mask in linear mode.
    pkg, vp = _render_pkg(with_linear=True), _viewpoint()
    vp.original_image[:, 1, 1] = RGB_BOUNDARY  # sum = 3 * thr > thr
    assert sRGB2Linear(vp.original_image[:, 1, 1]).sum() < RGB_BOUNDARY
    pkg["radiance_linear"] = pkg["radiance_linear"].clone().requires_grad_(True)
    get_loss_mapping(_config("linear", affine=False), pkg, vp).backward()
    assert pkg["radiance_linear"].grad[:, 1, 1].abs().sum() > 0
