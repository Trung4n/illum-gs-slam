"""light_models/diagnostics.py: albedo > 1 statistics and CSV log (needs torch)."""
import csv

import pytest

torch = pytest.importorskip("torch")

from light_models.diagnostics import CSV_NAME, albedo_stats, log_albedo_stats  # noqa: E402


def test_stats_count_gaussians_not_channels():
    albedo = torch.tensor(
        [[0.2, 0.3, 0.4], [1.5, 0.1, 0.1], [0.9, 1.2, 1.1], [1.0, 1.0, 1.0]]
    )
    s = albedo_stats(albedo)
    assert s["n_gaussians"] == 4
    assert s["frac_any_gt1"] == pytest.approx(0.5)  # exactly 1.0 is not above
    assert (s["frac_r_gt1"], s["frac_g_gt1"], s["frac_b_gt1"]) == pytest.approx(
        (0.25, 0.25, 0.25)
    )
    assert s["max"] == pytest.approx(1.5)


def test_empty_selection():
    assert albedo_stats(torch.zeros(0, 3)) == {"n_gaussians": 0}


def test_csv_appends_with_single_header(tmp_path):
    log_albedo_stats(str(tmp_path), 0, "new", torch.tensor([[1.2, 0.5, 0.5]]))
    log_albedo_stats(str(tmp_path), 0, "map", torch.tensor([[0.5, 0.5, 0.5]]))
    log_albedo_stats(str(tmp_path), 4, "new", torch.zeros(0, 3))
    with open(tmp_path / CSV_NAME) as f:
        rows = list(csv.DictReader(f))
    assert [(r["frame_idx"], r["stage"]) for r in rows] == [("0", "new"), ("0", "map"), ("4", "new")]
    assert float(rows[0]["frac_any_gt1"]) == 1.0
    assert float(rows[1]["frac_any_gt1"]) == 0.0
    assert rows[2]["n_gaussians"] == "0" and rows[2]["frac_any_gt1"] == ""


def test_no_file_without_save_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_albedo_stats(None, 0, "map", torch.tensor([[1.2, 0.5, 0.5]]))
    assert not (tmp_path / CSV_NAME).exists()
