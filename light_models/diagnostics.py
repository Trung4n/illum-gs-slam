# Run-time diagnostics of the light model. Logging only: nothing here feeds
# back into tracking or mapping.
#
# Albedo above 1 (docs/DECISIONS.md D1). The albedo is not clamped above 1 on
# purpose, so that a mismatch between the calibrated light intensity and the
# data shows up instead of being hidden: a large or growing fraction of
# Gaussians above 1 means the fixed intensity is too low for the observed
# images (or the Replica base color was baked too bright). The bound 1 is the
# definition of a physical albedo, not a tunable threshold.
#
# Two stages per keyframe, so the source of the excess can be told apart:
#   new - the Gaussians just created for this keyframe (Light.init_albedo);
#   map - every Gaussian after this keyframe's mapping.
# Printed with Log (no square brackets in the message: Log uses rich, which
# would swallow "[new]" / "[map]" as markup) and, when the run saves
# results, appended to
# <save_dir>/albedo_stats.csv.
#
# No torch import: only tensor methods of the input are used.

import math
import os

from utils.logging_utils import Log

CSV_NAME = "albedo_stats.csv"
_COLUMNS = (
    "frame_idx",
    "stage",
    "n_gaussians",
    "frac_any_gt1",
    "frac_r_gt1",
    "frac_g_gt1",
    "frac_b_gt1",
    "max",
    "mean",
)


def albedo_stats(albedo):
    """albedo: (N,3) tensor. Fractions are over Gaussians (not pixels)."""
    n = int(albedo.shape[0])
    if n == 0:
        return {"n_gaussians": 0}
    above = albedo > 1.0
    per_channel = above.float().mean(dim=0)
    return {
        "n_gaussians": n,
        "frac_any_gt1": above.any(dim=1).float().mean().item(),
        "frac_r_gt1": per_channel[0].item(),
        "frac_g_gt1": per_channel[1].item(),
        "frac_b_gt1": per_channel[2].item(),
        "max": albedo.max().item(),
        "mean": albedo.mean().item(),
    }


def log_albedo_stats(save_dir, frame_idx, stage, albedo):
    """Logs albedo_stats for one (keyframe, stage). save_dir None = no file
    (Results.save_results: false); the console line is always printed."""
    stats = albedo_stats(albedo.detach())
    row = {"frame_idx": frame_idx, "stage": stage, **stats}
    if stats["n_gaussians"] == 0:
        Log(f"albedo stage={stage} kf {frame_idx}: no Gaussians", tag="Light")
    else:
        Log(
            f"albedo stage={stage} kf {frame_idx}: "
            f"{stats['frac_any_gt1']:.2%} of {stats['n_gaussians']} Gaussians > 1 "
            f"(max {stats['max']:.3f}, mean {stats['mean']:.3f})",
            tag="Light",
        )
    if save_dir is None:
        return
    path = os.path.join(save_dir, CSV_NAME)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(_COLUMNS) + "\n")
        f.write(",".join(_format(row.get(c)) for c in _COLUMNS) + "\n")


EXPOSURE_CSV_NAME = "exposure_stats.csv"
_EXPOSURE_COLUMNS = ("frame_idx", "stage", "exposure_a", "exposure_b", "gain_exp_a")


def log_exposure(save_dir, frame_idx, stage, viewpoint):
    """Per-keyframe affine exposure (Camera.exposure_a/b), for EVERY run,
    baseline included: reading the values changes no computation.

    Why: with a co-located light the affine term can absorb the brightness
    change caused by the moving lamp (docs/DECISIONS.md D6, step 4.3). The
    stages mirror the albedo log: `new` = value tracked by the frontend when
    the keyframe arrives, `map` = after this keyframe's mapping.
    """
    a = float(viewpoint.exposure_a.detach().item())
    b = float(viewpoint.exposure_b.detach().item())
    gain = math.exp(a)
    Log(
        f"exposure stage={stage} kf {frame_idx}: a={a:+.4f} (gain {gain:.4f}) b={b:+.4f}",
        tag="Light",
    )
    if save_dir is None:
        return
    row = {"frame_idx": frame_idx, "stage": stage, "exposure_a": a,
           "exposure_b": b, "gain_exp_a": gain}
    path = os.path.join(save_dir, EXPOSURE_CSV_NAME)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(_EXPOSURE_COLUMNS) + "\n")
        f.write(",".join(_format(row[c]) for c in _EXPOSURE_COLUMNS) + "\n")


def _format(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
