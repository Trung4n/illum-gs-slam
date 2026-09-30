"""Where does the ATE come from? Compares the keyframe trajectories of several
runs beyond the single ATE RMSE (CLAUDE.md section 9, step 4.3: "ATE should
drop, especially along the optical axis").

    python tools/analysis/compare_trajectories.py \
        --run baseline=results/office0_lit/<t1> --run light=results/office0_lit/<t2> \
        [--window 30] [--out analysis_dir]

Reads <run>/plot/trj_final.json (written by utils/eval_utils.eval_ate:
camera-to-world poses of every keyframe, estimated and ground truth). For
each run:
  - global Sim(3) alignment (Umeyama with scale, as evo's correct_scale for
    monocular), ATE RMSE (cross-checked against plot/stats_final.json);
  - the aligned position error expressed in the GROUND-TRUTH camera frame:
    x = right, y = down, z = along the optical axis (OpenCV);
  - the error per range of frames (does it grow steadily or jump?);
  - scale drift: Sim(3) fitted on sliding windows of --window keyframes,
    local scale relative to the global one, and the local ATE after that
    local alignment (error that is NOT explained by drift);
  - RPE between consecutive keyframes (after the global scale), total and
    along the optical axis.
Writes per-keyframe CSVs and plots to --out when given. Reads results only.
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# One implementation shared with the end-of-run report printed by slam.py.
from utils.trajectory_analysis import analyze, frame_ranges, load_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="label=run_dir")
    ap.add_argument("--window", type=int, default=30, help="keyframes per local Sim(3) window")
    ap.add_argument("--range-step", type=int, default=250, help="frames per reporting range")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs = []
    for spec in args.run:
        label, run_dir = spec.split("=", 1)
        runs.append((label, load_run(run_dir)))
    ranges = frame_ranges(max(int(r[1][0].max()) for r in runs), args.range_step)

    results = []
    print(f"{'run':24s} {'kf':>4s} {'ATE':>7s} {'saved':>7s} {'x':>7s} {'y':>7s} "
          f"{'z(axis)':>8s} {'RPE':>7s} {'RPE_z':>7s} {'localATE':>8s} {'scale drift':>14s}")
    for label, (ids, est, gt, saved) in runs:
        a = analyze(ids, est, gt, args.window, ranges)
        results.append((label, a))
        saved_s = f"{saved:.4f}" if saved is not None else "-"
        print(f"{label:24s} {a['n_kf']:4d} {a['ate']:7.4f} {saved_s:>7s} {a['ate_x']:7.4f} "
              f"{a['ate_y']:7.4f} {a['ate_z_axis']:8.4f} {a['rpe']:7.4f} {a['rpe_z_axis']:7.4f} "
              f"{a['local_ate']:8.4f}   [{a['local_scale_rel_min']:.2f}, {a['local_scale_rel_max']:.2f}]")
    print("\nATE per frame range (total / along optical axis), m:")
    print(f"{'frames':>11s} " + " ".join(f"{label:>22s}" for label, _ in results))
    for k, (lo, hi) in enumerate(ranges):
        cells = []
        for _, a in results:
            _, _, n, tot, z = a["ranges"][k]
            cells.append(f"{tot:9.4f} / {z:8.4f}  " if n else f"{'-':>22s}")
        print(f"{lo:5d}-{hi:<5d} " + " ".join(cells))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for label, a in results:
            pk = a["per_kf"]
            with open(os.path.join(args.out, f"per_kf_{label}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(pk))
                for row in zip(*pk.values()):
                    w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            for label, a in results:
                pk = a["per_kf"]
                axes[0].plot(pk["frame"], pk["err"], label=label)
                axes[1].plot(pk["frame"], np.abs(pk["err_z_axis"]), label=label)
                axes[2].plot(pk["frame"], pk["local_scale_rel"], label=label)
            axes[0].set_ylabel("ATE per keyframe [m]")
            axes[1].set_ylabel("|error along optical axis| [m]")
            axes[2].set_ylabel("local / global scale")
            axes[2].set_xlabel("frame")
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "trajectory_errors.png"), dpi=110)
            print(f"\nplots and per-keyframe CSVs written to {args.out}")
        except ImportError:
            print(f"\nper-keyframe CSVs written to {args.out} (matplotlib missing: no plot)")


if __name__ == "__main__":
    main()
