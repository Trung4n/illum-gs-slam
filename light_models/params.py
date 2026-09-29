# ParamSpec: the single format of every numeric light-model parameter
# (CLAUDE.md section 4.5):
#
#   <name>:
#     source: params_file | value
#     key: <dotted JSON path>     # required iff source = params_file
#     value: <number or list>     # required iff source = value
#     learnable: true | false
#     lr: <number>                # required iff learnable
#     prior_weight: <number>      # required iff learnable; 0 = no penalty
#
# Nothing has a default. Keys that do not belong (e.g. lr on a fixed
# parameter, key with source: value) are errors as well, so a typo cannot
# silently change an ablation.
#
# The params file is the per-scene light_params.json (docs/DECISIONS.md D13,
# D17). Its keys are nested, hence dotted paths such as
# "t_CL_m.estimated_opencv".
#
# Parsing does not import torch; ParamSpec.tensor() does, lazily.

import json
import numbers
import os
from dataclasses import dataclass

from utils.config_utils import require_key

_SOURCES = ("params_file", "value")


@dataclass(frozen=True)
class ParamSpec:
    name: str  # dotted config path, for messages
    value: object  # resolved number or nested list of numbers
    source: str
    key: object  # JSON path or None
    learnable: bool
    lr: object  # float or None
    prior_weight: object  # float or None

    def tensor(self):
        """CPU float32 tensor of the value (requires_grad iff learnable).
        Kept on CPU so shaders pickle into spawned processes; the shader
        moves it to the data's device when it runs."""
        import torch

        t = torch.tensor(self.value, dtype=torch.float32)
        return t.requires_grad_(self.learnable)


def params_file_path(config):
    """Light.params_file resolved against Dataset.dataset_path (D17)."""
    light = require_key(config, "Light", "")
    rel = require_key(light, "params_file", "Light")
    root = require_key(require_key(config, "Dataset", ""), "dataset_path", "Dataset")
    return os.path.join(root, rel)


def load_params_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"light params file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_dotted(data, key, where):
    node = data
    walked = []
    for part in key.split("."):
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"{where}.key '{key}': '{'.'.join(walked)}' not found in the params file"
            )
        node = node[part]
    return node


def _check_numeric(value, where):
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{where}: expected a number or list of numbers, got {value!r}")
    if isinstance(value, numbers.Real):
        return
    if isinstance(value, (list, tuple)) and value:
        for v in value:
            _check_numeric(v, where)
        return
    raise TypeError(f"{where}: expected a number or list of numbers, got {value!r}")


def resolve_param(block, where, params_data):
    """Parses one ParamSpec block. params_data: the loaded params file (dict),
    or None when no params file is available (then source: params_file fails)."""
    if not isinstance(block, dict):
        raise TypeError(f"{where} must be a ParamSpec block, got {block!r}")
    source = require_key(block, "source", where)
    if source not in _SOURCES:
        raise ValueError(f"{where}.source must be one of {_SOURCES}, got {source!r}")
    learnable = require_key(block, "learnable", where)
    if not isinstance(learnable, bool):
        raise TypeError(f"{where}.learnable must be true or false, got {learnable!r}")

    allowed = {"source", "learnable"}
    if source == "params_file":
        key = require_key(block, "key", where)
        if not isinstance(key, str) or not key:
            raise TypeError(f"{where}.key must be a non-empty string, got {key!r}")
        if params_data is None:
            raise ValueError(f"{where}: source params_file but no params file was loaded")
        value = lookup_dotted(params_data, key, where)
        allowed.add("key")
    else:
        key = None
        value = require_key(block, "value", where)
        allowed.add("value")

    lr = prior_weight = None
    if learnable:
        lr = require_key(block, "lr", where)
        prior_weight = require_key(block, "prior_weight", where)
        for name, v in (("lr", lr), ("prior_weight", prior_weight)):
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or v < 0:
                raise ValueError(f"{where}.{name} must be a number >= 0, got {v!r}")
        allowed.update(("lr", "prior_weight"))

    extra = set(block) - allowed
    if extra:
        raise ValueError(
            f"{where}: unexpected keys {sorted(extra)} "
            f"(source={source}, learnable={learnable} allows {sorted(allowed)})"
        )
    _check_numeric(value, where)
    return ParamSpec(where, value, source, key, learnable, lr, prior_weight)
