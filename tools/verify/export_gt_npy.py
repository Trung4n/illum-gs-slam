"""Exports ground-truth passes of a few frames into a small, self-contained
bundle for the verification scripts (docs/DECISIONS.md D18, option c).

Runs in the environment of the DATASET repo (Replica), which can read EXR;
it imports nothing from MonoGS. The MonoGS side (tools/verify/*.py) then only
reads .npy/.png files, so environment.yml needs no EXR dependency, and the
verification calls the very light_models/ package SLAM uses.

    python tools/verify/export_gt_npy.py --replica-root D:/CodeForYou/Replica \
        --room office0 --frames 0 250 500 --out <bundle dir>

Bundle layout (one directory per frame):
    <out>/<room>/frame%06d/
        depth_m.npy     (H,W) float32  z-depth in meters (along the optical axis), 0 = invalid
        normal_cam.npy  (H,W,3) float32 unit normals in the OpenCV camera frame, 0 = invalid
        c2w.npy         (4,4) float64  camera-to-world pose (traj.txt row, OpenCV axes)
        albedo.png      albedo pass as rendered (8-bit sRGB)
        rgb_lit.png     lit image as rendered (8-bit sRGB PNG, not the JPEG SLAM reads)
    <out>/<room>/bundle.json   sources and per-frame statistics

Reading uses the dataset repo's own loaders (scripts/physics_check/common.py,
scripts/paths.py), i.e. exactly the conventions its physics check validated.
This is a GROUND-TRUTH consumer: it lives in tools/verify/ and must never be
imported by the SLAM path (CLAUDE.md section 3).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def _import_replica(replica_root):
    scripts = os.path.join(replica_root, "scripts")
    for p in (scripts, os.path.join(scripts, "physics_check")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import common  # noqa: E402  (Replica/scripts/physics_check/common.py)
    import paths  # noqa: E402  (Replica/scripts/paths.py)

    return common, paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replica-root", required=True)
    ap.add_argument("--room", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--render-dir",
        default=None,
        help="directory with rgb_png/ and depth_exr/ (default: the room's lit render)",
    )
    args = ap.parse_args(argv)

    common, paths = _import_replica(os.path.abspath(args.replica_root))
    rp = paths.room_paths(args.room)
    render_dir = os.path.abspath(args.render_dir or rp.render_lit)
    traj = common.read_traj(rp.traj)

    out_room = os.path.join(os.path.abspath(args.out), args.room)
    os.makedirs(out_room, exist_ok=True)
    meta = {
        "room": args.room,
        "render_dir": render_dir,
        "passes_dir": rp.render_passes,
        "traj": rp.traj,
        "depth_convention": "z-depth (optical axis), meters; 0 = invalid",
        "normal_convention": "unit, OpenCV camera frame (x right, y down, z forward); 0 = invalid",
        "frames": {},
    }
    for i in args.frames:
        d = os.path.join(out_room, "frame%06d" % i)
        os.makedirs(d, exist_ok=True)
        z, valid = common.load_depth(os.path.join(render_dir, "depth_exr", "depth%06d.exr" % i))
        n_world = common.load_normal_world(
            os.path.join(rp.render_passes, "normal_exr", "normal%06d.exr" % i)
        )
        c2w = traj[i]
        n = common.world_to_cam_normals(n_world, c2w)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n_ok = valid & np.isfinite(n).all(-1) & (norm[..., 0] > 0)
        n = np.where(n_ok[..., None], n / np.maximum(norm, 1e-30), 0.0)

        np.save(os.path.join(d, "depth_m.npy"), np.where(valid, z, 0.0).astype(np.float32))
        np.save(os.path.join(d, "normal_cam.npy"), n.astype(np.float32))
        np.save(os.path.join(d, "c2w.npy"), np.asarray(c2w, np.float64))
        shutil.copyfile(
            os.path.join(rp.render_passes, "albedo_png", "albedo%06d.png" % i),
            os.path.join(d, "albedo.png"),
        )
        shutil.copyfile(
            os.path.join(render_dir, "rgb_png", "frame%06d.png" % i),
            os.path.join(d, "rgb_lit.png"),
        )
        meta["frames"][str(i)] = {
            "valid_depth_px": int(valid.sum()),
            "valid_normal_px": int(n_ok.sum()),
            "depth_range_m": [float(z[valid].min()), float(z[valid].max())] if valid.any() else None,
        }
        print("frame %d: %d valid depth px, %d valid normal px" % (i, valid.sum(), n_ok.sum()))
    with open(os.path.join(out_room, "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("bundle written to", out_room)


if __name__ == "__main__":
    main()
