# Shaders: the image-space step applied by render() on top of the rasterized
# G-buffer (see gaussian_renderer.render for the call contract). Selected by
# name with Light.shader.type and looked up in _REGISTRY, never by if/else at
# call sites (CLAUDE.md section 2).
#
# Shader objects are handed to the backend and GUI processes, which are
# started with multiprocessing "spawn": they must be picklable, i.e.
# module-level classes, no closures or lambdas.
#
# This module must not import torch (the config/registry tests run without
# it); shaders only use tensor methods of their inputs.


class IdentityShader:
    """radiance_linear = albedo: unit shading, no light at all.

    Not a light model: the test mode of docs/DECISIONS.md D6. With
    loss_color_space: srgb it is original MonoGS with the color stored in
    linear space (render = linear2sRGB(albedo)), so its ATE must match the
    baseline within the configured tolerance; a larger gap points at the
    color path (storage, conversion, init), not at the light model.
    """

    def __call__(self, gbuffer, viewpoint_camera):
        return gbuffer["albedo"]


# type name -> (class, allowed parameter keys of Light.shader besides `type`)
_REGISTRY = {
    "identity": (IdentityShader, frozenset()),
}
# Names that are accepted by the config but not implemented yet.
PLANNED_SHADERS = ("colocated",)


def make_shader(shader_cfg):
    """Instantiates the shader described by the Light.shader block."""
    if not isinstance(shader_cfg, dict) or "type" not in shader_cfg:
        raise KeyError("Missing required config key 'Light.shader.type'")
    name = shader_cfg["type"]
    if name in PLANNED_SHADERS:
        raise NotImplementedError(
            f"Light.shader.type '{name}' is planned but not implemented yet"
        )
    if name not in _REGISTRY:
        raise ValueError(
            f"Light.shader.type must be one of "
            f"{tuple(_REGISTRY) + PLANNED_SHADERS}, got {name!r}"
        )
    cls, allowed = _REGISTRY[name]
    params = {k: v for k, v in shader_cfg.items() if k != "type"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in Light.shader for type '{name}' "
            f"(allowed: {sorted(allowed)})"
        )
    return cls(**params)
