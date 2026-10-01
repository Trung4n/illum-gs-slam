# Trajectory error analysis shared by slam.py (printed at the end of every
# run, so the numbers survive in the notebook log even if the result folder
# is lost) and tools/analysis/compare_trajectories.py (several runs side by
# side). numpy only. Reads the saved keyframe trajectory; never used by
# tracking or mapping.
#
# For a run: global alignment (Umeyama), ATE RMSE, the aligned position error
# in the ground-truth camera frame (x right, y down, z = optical axis), the
# error per range of frames, scale drift from sliding-window fits, the local
# ATE after that local alignment, and RPE between consecutive keyframes.
# The alignment follows utils/eval_utils.evaluate_evo: Sim(3) (evo's
# correct_scale) for monocular, SE(3) otherwise (with_scale=False); the
# local scale is then 1 by construction.

import json
import os

import numpy as np


def umeyama(src, dst, *, with_scale):
    """Sim(3) (s, R, t) minimizing ||dst - (s R src + t)||, src/dst (N,3);
    with_scale=False: SE(3), s = 1."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        s = np.trace(np.diag(d) @ sign) / var_s
    else:
        s = 1.0
    t = mu_d - s * r @ mu_s
    return s, r, t


def load_run(run_dir):
    with open(os.path.join(run_dir, "plot", "trj_final.json"), "r") as f:
        trj = json.load(f)
    est = np.asarray(trj["trj_est"], dtype=np.float64)
    gt = np.asarray(trj["trj_gt"], dtype=np.float64)
    ids = np.asarray(trj["trj_id"])
    stats_path = os.path.join(run_dir, "plot", "stats_final.json")
    rmse_saved = None
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            rmse_saved = json.load(f).get("rmse")
    return ids, est, gt, rmse_saved


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def alignment_name(with_scale):
    return "Sim(3)" if with_scale else "SE(3)"


def analyze(ids, est, gt, window, ranges, *, with_scale):
    p_est, p_gt = est[:, :3, 3], gt[:, :3, 3]
    s, r, t = umeyama(p_est, p_gt, with_scale=with_scale)
    aligned = (s * (r @ p_est.T)).T + t
    err = aligned - p_gt
    # Error in the ground-truth camera frame: e_cam = R_gt^T e (c2w rotations).
    err_cam = np.einsum("nji,nj->ni", gt[:, :3, :3], err)
    out = {
        "n_kf": len(ids),
        "global_scale": float(s),
        "ate": rmse(np.linalg.norm(err, axis=1)),
        "ate_x": rmse(err_cam[:, 0]),
        "ate_y": rmse(err_cam[:, 1]),
        "ate_z_axis": rmse(err_cam[:, 2]),
    }
    # Sliding-window alignment: local scale and drift-free local error.
    local_scale = np.full(len(ids), np.nan)
    local_err = np.full(len(ids), np.nan)
    half = window // 2
    for i in range(len(ids)):
        lo, hi = max(0, i - half), min(len(ids), i + half + 1)
        if hi - lo < max(4, window // 2):
            continue
        sl, rl, tl = umeyama(p_est[lo:hi], p_gt[lo:hi], with_scale=with_scale)
        local_scale[i] = sl / s
        local_err[i] = np.linalg.norm(sl * rl @ p_est[i] + tl - p_gt[i])
    # Ground-truth motion between consecutive keyframes: translation and
    # rotation angle. Monocular scale is unobservable when the camera mostly
    # rotates in place, the usual suspect for local scale excursions.
    step_t = np.linalg.norm(np.diff(p_gt, axis=0), axis=1)
    rel_r = np.einsum("nji,njk->nik", gt[:-1, :3, :3], gt[1:, :3, :3])
    step_rot = np.degrees(np.arccos(np.clip((np.trace(rel_r, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out["ranges"] = []
    for lo, hi in ranges:
        m = (ids >= lo) & (ids < hi)
        ms = m[1:] & m[:-1]  # steps whose two keyframes are both in range
        ls = local_scale[m]
        ls = ls[~np.isnan(ls)]
        out["ranges"].append({
            "lo": lo, "hi": hi, "n": int(m.sum()),
            "ate": rmse(np.linalg.norm(err[m], axis=1)),
            "ate_z_axis": rmse(err_cam[m, 2]),
            "local_ate": rmse(local_err[m][~np.isnan(local_err[m])]),
            "local_scale_min": float(ls.min()) if len(ls) else float("nan"),
            "local_scale_max": float(ls.max()) if len(ls) else float("nan"),
            "gt_path_m": float(step_t[ms].sum()),
            "gt_rot_deg": float(step_rot[ms].sum()),
        })
    out["local_scale_rel_min"] = float(np.nanmin(local_scale))
    out["local_scale_rel_max"] = float(np.nanmax(local_scale))
    out["local_ate"] = rmse(local_err[~np.isnan(local_err)])
    # RPE between consecutive keyframes (global scale applied to estimates).
    d_est = s * (r @ np.diff(p_est, axis=0).T).T
    d_gt = np.diff(p_gt, axis=0)
    rpe = d_est - d_gt
    rpe_cam = np.einsum("nji,nj->ni", gt[:-1, :3, :3], rpe)
    out["rpe"] = rmse(np.linalg.norm(rpe, axis=1))
    out["rpe_z_axis"] = rmse(rpe_cam[:, 2])
    out["per_kf"] = {
        "frame": ids,
        "err": np.linalg.norm(err, axis=1),
        "err_x": err_cam[:, 0],
        "err_y": err_cam[:, 1],
        "err_z_axis": err_cam[:, 2],
        "local_scale_rel": local_scale,
        "local_err": local_err,
    }
    return out


def frame_ranges(last_frame, step):
    return [(lo, lo + step) for lo in range(0, last_frame + 1, step)]


def run_report(run_dir, window, range_step, *, with_scale):
    """Text lines summarizing one run's plot/trj_final.json."""
    ids, est, gt, saved = load_run(run_dir)
    a = analyze(ids, est, gt, window, frame_ranges(int(ids.max()), range_step),
                with_scale=with_scale)
    saved_s = f"{saved:.4f}" if saved is not None else "-"
    align = alignment_name(with_scale)
    lines = [
        f"trajectory analysis: {a['n_kf']} keyframes, global {align} scale {a['global_scale']:.4f}",
        f"  ATE {a['ate']:.4f} m (saved {saved_s}) | in camera frame: x {a['ate_x']:.4f}, "
        f"y {a['ate_y']:.4f}, z (optical axis) {a['ate_z_axis']:.4f}",
        f"  RPE {a['rpe']:.4f} m (optical axis {a['rpe_z_axis']:.4f}) | local ATE after "
        f"{window}-keyframe {align} {a['local_ate']:.4f} | local/global scale "
        f"[{a['local_scale_rel_min']:.2f}, {a['local_scale_rel_max']:.2f}]",
        "  per frame range: ATE total / optical axis | local ATE | local/global scale "
        "| ground-truth path (m), rotation (deg), rotation per m:",
    ]
    for g in a["ranges"]:
        if g["n"]:
            per_m = g["gt_rot_deg"] / g["gt_path_m"] if g["gt_path_m"] > 0 else float("inf")
            lines.append(
                f"    {g['lo']:5d}-{g['hi']:<5d} n={g['n']:3d}  {g['ate']:.4f} / {g['ate_z_axis']:.4f}"
                f" | {g['local_ate']:.4f} | [{g['local_scale_min']:.2f}, {g['local_scale_max']:.2f}]"
                f" | {g['gt_path_m']:.2f} m, {g['gt_rot_deg']:.0f} deg, {per_m:.0f} deg/m"
            )
    return lines
