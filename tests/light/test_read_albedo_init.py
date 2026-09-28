"""light_models.read_albedo_init: strict reading of Light.init_albedo."""
import pytest

from light_models import ALBEDO_INIT_STRATEGIES, read_albedo_init


def _config(block=None):
    light = {"enabled": True}
    if block is not None:
        light["init_albedo"] = block
    return {"Light": light}


def test_missing_block_raises():
    with pytest.raises(KeyError, match="Light.init_albedo"):
        read_albedo_init(_config())


def test_missing_strategy_raises():
    with pytest.raises(KeyError, match="Light.init_albedo.strategy"):
        read_albedo_init(_config({}))


@pytest.mark.parametrize("name", ["Observed", "median", "", None])
def test_unknown_strategy_raises(name):
    with pytest.raises(ValueError, match="strategy"):
        read_albedo_init(_config({"strategy": name}))


@pytest.mark.parametrize("name", ALBEDO_INIT_STRATEGIES)
def test_every_planned_name_parses(name):
    assert read_albedo_init(_config({"strategy": name})).strategy == name


def test_params_are_the_other_keys():
    s = read_albedo_init(_config({"strategy": "rendered_depth", "opacity_thr": 0.5}))
    assert s.params == {"opacity_thr": 0.5}
