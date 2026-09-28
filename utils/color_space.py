# The ONLY place in the SLAM path that converts between sRGB-encoded values
# and linear radiance/albedo (CLAUDE.md section 6: sRGB conversion lives in
# exactly one place). Every other module imports these two functions; nobody
# re-implements the transfer curve inline.
#
# Convention: the standard sRGB transfer functions of IEC 61966-2-1, exactly
# the ones the dataset repo uses to verify the lighting model
# (Replica/scripts/physics_check/common.py: srgb_to_lin, lin_to_srgb8), so
# that "linear" here means the same thing as "linear" in light_params.json.
# The numbers below are fixed by the standard, not tunable parameters of the
# light model, which is why they live in code and not in the config.
#
# Differences from the dataset repo, both deliberate:
#   - Inputs/outputs are float tensors in [0, 1] (the dataset loader already
#     divides by 255), not 0-255 integers.
#   - linear2sRGB does NOT clip to [0, 1] and does not round. Blender clips
#     when writing the PNG, but here the output feeds a loss: clipping would
#     zero the gradient of every over-exposed pixel. Callers that need a
#     displayable image clamp themselves (eval_utils/GUI already do).
#
# Both functions are differentiable everywhere. torch.where evaluates BOTH
# branches, so the power branch gets its input clamped to the knee first:
# otherwise a negative input would give NaN in the unused branch and the NaN
# would leak into the gradient.

import torch

# EOTF (sRGB-encoded -> linear): linear segment below this encoded value.
_EOTF_KNEE = 0.04045
# OETF (linear -> sRGB-encoded): linear segment below this linear value.
_OETF_KNEE = 0.0031308
_LINEAR_SLOPE = 12.92
_A = 0.055
_GAMMA = 2.4


def sRGB2Linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB-encoded values in [0, 1] -> linear values (IEC 61966-2-1 EOTF)."""
    low = x / _LINEAR_SLOPE
    high = ((torch.clamp_min(x, _EOTF_KNEE) + _A) / (1.0 + _A)) ** _GAMMA
    return torch.where(x <= _EOTF_KNEE, low, high)


def linear2sRGB(x: torch.Tensor) -> torch.Tensor:
    """Linear values -> sRGB-encoded values (IEC 61966-2-1 OETF), no clipping."""
    low = _LINEAR_SLOPE * x
    high = (1.0 + _A) * torch.clamp_min(x, _OETF_KNEE) ** (1.0 / _GAMMA) - _A
    return torch.where(x <= _OETF_KNEE, low, high)
