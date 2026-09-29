# Falloff(d_l): attenuation with the distance d_l from the surface point to
# the LIGHT (not to the camera: the light sits at t_CL, see colocated.py).
# d_l > 0 is guaranteed by the caller (points are in front of the camera and
# invalid pixels get a finite fallback depth), so no epsilon here.

from light_models.registry import declare_planned, register


@register("falloff", "inverse_square")
class InverseSquare:
    """1 / d^2 (verified on the simulated data: log-log slope ~ -2)."""

    def __call__(self, dist):
        return 1.0 / (dist * dist)


@register("falloff", "lorentzian", numeric=("tau",))
class Lorentzian:
    """1 / (d^2 + tau^2): inverse square that stays finite near the source."""

    def __init__(self, tau):
        if not isinstance(tau.value, (int, float)) or tau.value <= 0:
            raise ValueError(f"{tau.name} must be a number > 0, got {tau.value!r}")
        self.tau = tau.tensor()

    def __call__(self, dist):
        tau = self.tau.to(dist.device, dist.dtype)
        return 1.0 / (dist * dist + tau * tau)


declare_planned("falloff", "learned")
