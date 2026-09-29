"""utils/slam_utils.saturated_observations (CPU, needs torch)."""
import pytest

torch = pytest.importorskip("torch")

from utils.slam_utils import saturated_observations  # noqa: E402


def _image(codes):
    # As the dataset loader stores it: 8-bit codes / 255, float32.
    return (torch.tensor(codes, dtype=torch.float64) / 255.0).float()


def test_code_at_threshold_counts_despite_float_roundoff():
    # 250/255 in float32 times 255 is not exactly 250; rounding to the code
    # must still put it at the threshold.
    img = _image([[[250.0]], [[0.0]], [[0.0]]])
    assert saturated_observations(img, "channel", 250)[0, 0, 0]
    assert not saturated_observations(_image([[[249.0]], [[0.0]], [[0.0]]]), "channel", 250).any()


def test_every_code_is_classified_by_its_integer_value():
    codes = torch.arange(256, dtype=torch.float64).view(1, 1, 256).expand(3, 1, 256)
    sat = saturated_observations((codes / 255.0).float(), "channel", 250)
    assert torch.equal(sat[0, 0], torch.arange(256) >= 250)


def test_pixel_mode_drops_all_channels_channel_mode_only_one():
    img = _image([[[255.0, 10.0]], [[10.0, 10.0]], [[10.0, 10.0]]])  # pixel 0: R saturated
    pix = saturated_observations(img, "pixel", 250)
    ch = saturated_observations(img, "channel", 250)
    assert pix.shape == (1, 1, 2) and pix[0, 0].tolist() == [True, False]
    assert ch.shape == (3, 1, 2) and ch[:, 0, 0].tolist() == [True, False, False]
