"""Albedo round-trip check (step 4.1): ground-truth albedo -> Gaussians ->
rasterized albedo map -> compare with the ground-truth albedo.

    python tools/verify/albedo_roundtrip.py --config configs/light/verify_roundtrip.yaml \
        --bundle <export dir>/<room> --frames 0 500 [--out result.json]

The bundle comes from tools/verify/export_gt_npy.py. Gaussians are created
through the SAME code SLAM uses (GaussianModel.extend_from_pcd_seq with an
albedo_map, render() with the configured shader), at the ground-truth pose,
from the ground-truth depth. No optimization, so no SLAM randomness.

Reported per frame, over pixels with opacity > Verify.roundtrip.opacity_thr:
  - |A_rec - A_gt| (linear), A_rec = rasterized albedo / opacity (D5);
  - the same error for two deliberately WRONG readings, which must be much
    larger if the check has any power: the albedo compared in sRGB
    (a skipped gamma) and without dividing by opacity.
Needs CUDA (rasterizer, simple_knn). Reads ground truth: verification only.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from munch import munchify  # noqa: E402
from PIL import Image  # noqa: E402

from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_splatting.utils.graphics_utils import focal2fov  # noqa: E402
from light_models import build_shader  # noqa: E402
from utils.camera_utils import Camera, build_projection_matrix  # noqa: E402
from utils.color_space import sRGB2Linear  # noqa: E402
from utils.config_utils import load_config, require_key  # noqa: E402


def load_png01(path, device):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().to(device)


def make_camera(config, c2w, color, uid, device):
    calib = config["Dataset"]["Calibration"]
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    w, h = calib["width"], calib["height"]
    # Same world->camera convention as ReplicaParser (inverse of traj.txt).
    gt_T = torch.from_numpy(np.linalg.inv(c2w)).to(device)
    cam = Camera(
        uid,
        color,
        None,
        gt_T,
        build_projection_matrix(fx, fy, cx, cy, w, h, device),
        fx,
        fy,
        cx,
        cy,
        focal2fov(fx, w),
        focal2fov(fy, h),
        h,
        w,
        device=device,
    )
    cam.update_RT(cam.R_gt, cam.T_gt)
    return cam


def _stats(err):
    return {
        "median": float(err.median()),
        "p95": float(torch.quantile(err, 0.95)),
        "mean": float(err.mean()),
    }


def check_frame(config, frame_dir, uid, device):
    opacity_thr = require_key(
        require_key(require_key(config, "Verify", ""), "roundtrip", "Verify"),
        "opacity_thr",
        "Verify.roundtrip",
    )
    depth = np.load(os.path.join(frame_dir, "depth_m.npy"))
    c2w = np.load(os.path.join(frame_dir, "c2w.npy"))
    albedo_srgb = load_png01(os.path.join(frame_dir, "albedo.png"), device)
    albedo_lin = sRGB2Linear(albedo_srgb)

    cam = make_camera(config, c2w, albedo_srgb, uid, device)
    gaussians = GaussianModel(0, config=config)
    gaussians.init_lr(6.0)  # as slam.py; learning rates are irrelevant here
    gaussians.training_setup(munchify(config["opt_params"]))
    gaussians.extend_from_pcd_seq(
        cam, kf_id=uid, init=True, depthmap=depth, albedo_map=albedo_lin
    )
    pipe = munchify(config["pipeline_params"])
    background = torch.tensor(
        config["pipeline_params"]["background"], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, background, shader=build_shader(config))
    albedo_w = pkg["albedo"]  # opacity-weighted, linear
    opacity = pkg["opacity"]

    mask = (opacity[0] > opacity_thr) & torch.from_numpy(depth > 0).to(device)
    a_rec = albedo_w / opacity.clamp_min(opacity_thr)
    sel = mask[None].expand_as(a_rec)
    err = (a_rec - albedo_lin).abs()[sel]
    err_gamma = (a_rec - albedo_srgb).abs()[sel]
    err_premult = (albedo_w - albedo_lin).abs()[sel]
    ratio = (a_rec.sum(0)[mask] / albedo_lin.sum(0)[mask].clamp_min(1e-6)).median()
    return {
        "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "coverage": float(mask.float().mean()),
        "opacity_median": float(opacity[0][mask].median()),
        "abs_err_linear": _stats(err),
        "abs_err_if_gamma_skipped": _stats(err_gamma),
        "abs_err_if_not_divided_by_opacity": _stats(err_premult),
        "median_ratio_rec_over_gt": float(ratio),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--bundle", required=True, help="<export dir>/<room>")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None, help="optional JSON file for the results")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("albedo_roundtrip.py needs CUDA (rasterizer)")
    config = load_config(args.config)
    results = {}
    for i in args.frames:
        r = check_frame(config, os.path.join(args.bundle, "frame%06d" % i), i, "cuda")
        results[i] = r
        print(
            f"frame {i}: coverage {r['coverage']:.1%}, opacity median {r['opacity_median']:.3f}\n"
            f"  |A_rec - A_gt| linear       median {r['abs_err_linear']['median']:.4f}"
            f"  p95 {r['abs_err_linear']['p95']:.4f}\n"
            f"  if gamma were skipped       median {r['abs_err_if_gamma_skipped']['median']:.4f}\n"
            f"  if not divided by opacity   median {r['abs_err_if_not_divided_by_opacity']['median']:.4f}\n"
            f"  median A_rec/A_gt           {r['median_ratio_rec_over_gt']:.4f}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
