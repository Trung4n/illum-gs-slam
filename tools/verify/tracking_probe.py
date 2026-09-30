"""Tracking probe (docs/DECISIONS.md D39): can SLAM's own tracking localize a
frame against a PERFECT map? Separates tracking from mapping.

    python tools/verify/tracking_probe.py --config <experiment yaml> \
        --probe configs/light/verify_tracking.yaml \
        --dataset_path <scene dir> --bundle <export dir>/<room>

For each reference frame r of the probe config, a map is built from the
ground-truth depth of r, at r's ground-truth pose, one Gaussian per pixel,
with the opacity of the probe config:
  - experiment with Light.enabled: false -> colors = the observed image of r
    (lighting baked in: the best map original MonoGS could learn);
  - experiment with a light model -> albedo = the ground-truth albedo pass,
    shaded by the experiment's shader.
Then FrontEnd.tracking (the unchanged SLAM code, with the experiment's
losses, masks and exposure setting) tracks frame r + offset starting from
r's pose, like SLAM starts from the previous frame. Reported: position error
(cm; its component along the target camera's optical axis) and rotation
error (deg), before (= the motion to recover) and after tracking.

No scale ambiguity here: the map is metric. Needs CUDA. Reads ground truth:
verification only, never imported by the SLAM path.
"""
import argparse
import copy
import math
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
from light_models import build_shader  # noqa: E402
from utils.camera_utils import Camera, build_projection_matrix  # noqa: E402
from utils.color_space import sRGB2Linear  # noqa: E402
from utils.config_utils import load_config, require_key  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402
from utils.multiprocessing_utils import FakeQueue  # noqa: E402
from utils.slam_frontend import FrontEnd  # noqa: E402


def c2w(cam):
    w2c = np.eye(4)
    w2c[:3, :3] = cam.R.detach().cpu().numpy()
    w2c[:3, 3] = cam.T.detach().cpu().numpy()
    return np.linalg.inv(w2c)


def c2w_gt(cam):
    w2c = np.eye(4)
    w2c[:3, :3] = cam.R_gt.detach().cpu().numpy()
    w2c[:3, 3] = cam.T_gt.detach().cpu().numpy()
    return np.linalg.inv(w2c)


def pose_error(est, gt):
    """(position error m, its component along gt's optical axis m, rotation deg)."""
    d = est[:3, 3] - gt[:3, 3]
    along = float(d @ gt[:3, 2])
    r = est[:3, :3].T @ gt[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(d)), along, float(ang)


def build_map(config, probe, dataset, proj, ref, bundle, shader):
    cfg = copy.deepcopy(config)
    cfg["Dataset"]["pcd_downsample_init"] = probe["pcd_downsample"]
    cam = Camera.init_from_dataset(dataset, ref, proj)
    cam.update_RT(cam.R_gt, cam.T_gt)
    frame_dir = os.path.join(bundle, "frame%06d" % ref)
    depth = np.load(os.path.join(frame_dir, "depth_m.npy")).astype(np.float32)
    albedo_map = None
    if shader is not None:
        alb = np.asarray(Image.open(os.path.join(frame_dir, "albedo.png")).convert("RGB"),
                         dtype=np.float32) / 255.0
        albedo_map = sRGB2Linear(torch.from_numpy(alb).permute(2, 0, 1).contiguous()).cuda()
    gaussians = GaussianModel(0, config=cfg)
    gaussians.init_lr(6.0)  # as slam.py; the map is not optimized here
    gaussians.training_setup(munchify(cfg["opt_params"]))
    gaussians.extend_from_pcd_seq(
        cam, kf_id=ref, init=True, depthmap=depth, albedo_map=albedo_map
    )
    with torch.no_grad():
        gaussians._opacity.fill_(float(torch.logit(torch.tensor(probe["gaussian_opacity"]))))
        # Enlarge the Gaussians so the map covers every pixel like a converged
        # SLAM map: built on the data's pixel grid they sit half a pixel off
        # the rasterizer's (D29) and at their creation size cover a pixel only
        # ~0.95, i.e. about half the pixels fall below the 0.95 trust
        # threshold of the light model's G-buffer (D41).
        gaussians._scaling.add_(math.log(probe["gaussian_scale_mult"]))
    return gaussians, cam


def coverage_line(gaussians, cam, config, shader, background):
    with torch.no_grad():
        pkg = render(cam, gaussians, munchify(config["pipeline_params"]), background, shader=shader)
    op = pkg["opacity"][0].flatten()
    line = (f"     map coverage at the reference pose: opacity p5 {op.quantile(0.05):.3f}, "
            f"median {op.median():.3f}")
    thr = getattr(shader, "opacity_thr", None)
    if thr is not None:
        line += f", trusted by the shader (>= {thr}) {(op >= thr).float().mean():.1%}"
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="experiment config")
    ap.add_argument("--probe", required=True, help="configs/light/verify_tracking.yaml")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--bundle", required=True, help="<export dir>/<room>")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("tracking_probe.py needs CUDA")

    config = load_config(args.config)
    config["Dataset"]["dataset_path"] = args.dataset_path
    config["Training"]["monocular"] = config["Dataset"]["sensor_type"] == "monocular"
    config["Results"]["save_dir"] = None
    probe = require_key(require_key(load_config(args.probe), "Verify", ""), "tracking", "Verify")
    for key in ("gaussian_opacity", "gaussian_scale_mult", "pcd_downsample", "ref_frames", "offsets"):
        require_key(probe, key, "Verify.tracking")

    model_params = munchify(config["model_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    proj = build_projection_matrix(
        dataset.fx, dataset.fy, dataset.cx, dataset.cy, dataset.width, dataset.height, "cuda"
    )
    shader = build_shader(config)
    background = torch.tensor(
        config["pipeline_params"]["background"], dtype=torch.float32, device="cuda"
    )
    mode = "baked colors (light off)" if shader is None else f"light model ({type(shader).__name__})"
    print(f"config {args.config}: {mode}")
    print(" ref  +off | before: pos cm (axis cm)  rot deg | after: pos cm (axis cm)  rot deg")

    rows = []
    for ref in probe["ref_frames"]:
        gaussians, ref_cam = build_map(config, probe, dataset, proj, ref, args.bundle, shader)
        print(coverage_line(gaussians, ref_cam, config, shader, background))
        for off in probe["offsets"]:
            tgt = ref + off
            if tgt >= len(dataset):
                continue
            fe = FrontEnd(config)
            fe.set_hyperparams()
            fe.dataset, fe.background, fe.shader = dataset, background, shader
            fe.pipeline_params = munchify(config["pipeline_params"])
            fe.q_main2vis = FakeQueue()
            fe.gaussians = gaussians
            fe.use_every_n_frames = off
            fe.cameras = {ref: ref_cam}
            view = Camera.init_from_dataset(dataset, tgt, proj)
            view.compute_grad_mask(config)
            gt = c2w_gt(view)
            before = pose_error(c2w_gt(ref_cam), gt)
            fe.tracking(tgt, view)
            after = pose_error(c2w(view), gt)
            rows.append((before, after))
            print(f"{ref:4d}  +{off:<3d} | {100*before[0]:6.2f} ({100*before[1]:+6.2f})  {before[2]:6.2f}"
                  f"   | {100*after[0]:6.2f} ({100*after[1]:+6.2f})  {after[2]:6.2f}")
        del gaussians
        torch.cuda.empty_cache()

    b = np.array([r[0] for r in rows])
    a = np.array([r[1] for r in rows])
    print(f"mean      | {100*b[:,0].mean():6.2f} (|axis| {100*np.abs(b[:,1]).mean():5.2f})  {b[:,2].mean():6.2f}"
          f"   | {100*a[:,0].mean():6.2f} (|axis| {100*np.abs(a[:,1]).mean():5.2f})  {a[:,2].mean():6.2f}")


if __name__ == "__main__":
    main()
