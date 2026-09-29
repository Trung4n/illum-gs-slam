# Cosine(n, l): the foreshortening term of the direct light.
# Each form returns (factor (H,W), aux) where aux holds per-pixel maps that
# the loss may use (pixel_weight.min_cos_nl, docs/DECISIONS.md D15).

import torch

from light_models.registry import build_component, declare_planned, register


@register("cosine", "none")
class NoCosine:
    """Factor 1: the direct term ignores surface orientation."""

    def __call__(self, points, valid, to_light, gbuffer):
        return torch.ones_like(valid, dtype=points.dtype), {}


@register("cosine", "lambert", options=("normal_source",))
class Lambert:
    """max(0, n . l), l = unit vector from the point to the light. Pixels
    without a valid normal get factor 0 and are reported in aux so the loss
    can exclude them (they would otherwise be shaded ambient-only)."""

    def __init__(self, normal_source):
        if not isinstance(normal_source, str):
            raise TypeError(f"Light.cosine.normal_source must be a name, got {normal_source!r}")
        self.normals = build_component(
            "normals", {"type": normal_source}, "Light.cosine.normal_source", None
        )

    def __call__(self, points, valid, to_light, gbuffer):
        n, n_valid = self.normals(points, valid, gbuffer)
        cos_nl = (n * to_light).sum(0)
        cos_nl = torch.where(n_valid, cos_nl, torch.zeros_like(cos_nl))
        return cos_nl.clamp_min(0.0), {"cos_nl": cos_nl, "normal_valid": n_valid}


declare_planned("cosine")  # no planned forms beyond none / lambert
