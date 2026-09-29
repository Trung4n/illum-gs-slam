# Pixel-index carrier for Open3D back-projection (docs/DECISIONS.md D28).
# GaussianModel.create_pcd_from_image gives Open3D this image instead of the
# RGB image when the light model supplies an albedo map, so each new point
# comes back labelled with the pixel it was back-projected from.
# numpy only: unit-testable without CUDA.

import numpy as np


def pixel_index_image(height, width):
    # (H,W,3) uint8 image whose "color" is the pixel index v * W + u in
    # base 256 (24 bits). Given to Open3D in place of the RGB image, it
    # survives back-projection and random_down_sample untouched, so every
    # point carries the exact pixel it came from. Earlier the pixel was
    # recovered by re-projecting each point, which failed on a SLAM run
    # (a point 0.446 px off the grid, 2026-09-29).
    if height * width > 1 << 24:
        raise ValueError(f"{height}x{width} pixels do not fit a 24-bit index")
    idx = np.arange(height * width, dtype=np.int64).reshape(height, width)
    code = np.stack([(idx >> 16) & 255, (idx >> 8) & 255, idx & 255], axis=-1)
    return np.ascontiguousarray(code.astype(np.uint8))


def decode_pixel_index(colors):
    # Inverse of pixel_index_image on Open3D's output colors, which are
    # the uint8 values divided by 255.0 (float64). Exact: decoding and
    # re-encoding must give back Open3D's values bit for bit.
    colors = np.asarray(colors, dtype=np.float64)
    code = np.rint(colors * 255.0).astype(np.int64)
    if not np.array_equal(code / 255.0, colors):
        raise RuntimeError("Open3D colors are not 8-bit pixel-index codes")
    return (code[:, 0] << 16) | (code[:, 1] << 8) | code[:, 2]
