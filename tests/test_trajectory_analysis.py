"""utils/trajectory_analysis.py: alignment matches eval_ate (Sim(3) for
monocular, SE(3) otherwise). numpy only.

    python -m pytest tests/test_trajectory_analysis.py
"""
import numpy as np
import pytest

from utils.trajectory_analysis import analyze, frame_ranges, umeyama


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _poses(p):
    t = np.tile(np.eye(4), (len(p), 1, 1))
    t[:, :3, 3] = p
    return t


def test_se3_keeps_scale_one_and_sim3_recovers_it():
    src = np.random.default_rng(0).normal(size=(50, 3))
    r, t = _rot(1), np.array([0.3, -1.0, 2.0])
    dst = (2.5 * (r @ src.T)).T + t
    s, rr, tt = umeyama(src, dst, with_scale=True)
    assert s == pytest.approx(2.5) and np.allclose(rr, r) and np.allclose(tt, t)
    s, rr, _ = umeyama(src, dst, with_scale=False)
    assert s == 1.0 and np.allclose(rr, r)


def test_rigid_trajectory_has_zero_ate_with_se3():
    p_est = np.cumsum(np.random.default_rng(2).normal(size=(40, 3)) * 0.1, axis=0)
    r, t = _rot(3), np.array([1.0, 2.0, 3.0])
    p_gt = (r @ p_est.T).T + t
    ids = np.arange(40) * 5
    a = analyze(ids, _poses(p_est), _poses(p_gt), 10, frame_ranges(int(ids.max()), 100),
                with_scale=False)
    assert a["global_scale"] == 1.0 and a["ate"] < 1e-9
    # A scaled copy is NOT explained by SE(3), only by Sim(3).
    scaled = _poses(2.0 * p_gt)
    assert analyze(ids, _poses(p_est), scaled, 10, [], with_scale=False)["ate"] > 0.1
    assert analyze(ids, _poses(p_est), scaled, 10, [], with_scale=True)["ate"] < 1e-9
