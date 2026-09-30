"""light_models.read_min_gradient: strict reading of LightTracking.min_gradient."""
import pytest

from light_models import read_min_gradient


def _cfg(block=None):
    lt = {} if block is None else {"min_gradient": block}
    return {"LightTracking": lt}


def test_block_required():
    with pytest.raises(KeyError, match="LightTracking.min_gradient"):
        read_min_gradient(_cfg())


def test_disabled():
    assert read_min_gradient(_cfg({"enabled": False})).enabled is False


def test_enabled():
    mg = read_min_gradient(_cfg({"enabled": True, "threshold_8bit": 2}))
    assert mg.enabled and mg.threshold_8bit == 2.0


def test_threshold_required_when_enabled():
    with pytest.raises(KeyError, match="min_gradient.threshold_8bit"):
        read_min_gradient(_cfg({"enabled": True}))


@pytest.mark.parametrize("value", [0, -1, "2", True, None])
def test_threshold_validated(value):
    with pytest.raises(ValueError, match="threshold_8bit"):
        read_min_gradient(_cfg({"enabled": True, "threshold_8bit": value}))


def test_unknown_key():
    with pytest.raises(ValueError, match="unexpected keys"):
        read_min_gradient(_cfg({"enabled": False, "relative": 4}))
