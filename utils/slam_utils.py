import torch

# Loss functions and image helpers shared by tracking (FrontEnd) and mapping
# (BackEnd). All losses are plain masked L1 photometric (+ optional depth)
# errors between a rendered frame and the frame's GT image/depth; the masks
# are what differ between tracking vs. mapping and monocular vs. RGB-D.
#
# Shapes used throughout: images are (3,H,W), depth/opacity maps are
# (1,H,W), all on CUDA.


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    # Returns per-channel (vertical, horizontal) derivatives, same (C,H,W)
    # shape as the input thanks to the 1-pixel reflect padding.
    # NOTE on naming: the kernel called `conv_y` actually differentiates along
    # x (columns) and produces img_grad_h; `conv_x` differentiates along y
    # (rows) and produces img_grad_v. The returned v/h names are the ones to
    # trust.
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    # 1/32: sum of |kernel weights|, keeps gradient magnitudes in the same
    # range as the input intensities.
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    # groups=c: filter each channel independently (depthwise convolution).
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    # Marks pixels whose ENTIRE 3x3 neighbourhood has |value| > eps, i.e.
    # pixels where a 3x3 gradient can be trusted. A gradient computed next to
    # an invalid (zero/near-black) pixel — image border, black undistortion
    # margin, missing depth — would be a fake edge; this mask removes those.
    # Box-filtering the boolean map and checking "== 9" is a cheap way to ask
    # "are all 9 neighbours valid?".
    # The two returned masks are identical (both kernels are all-ones); two
    # are returned only to mirror image_gradient()'s (v, h) signature.
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    # Edge-aware depth smoothness regularizer: penalizes depth gradients,
    # but down-weights the penalty where the color image itself has strong
    # edges (w = exp(-10 * grad^2)), so depth is allowed to jump at object
    # boundaries and forced to be smooth inside textureless surfaces.
    # NOT CALLED anywhere in this codebase (and `huber_eps`/`mask` are
    # unused) — an unused experiment kept from development.
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    # Entry point for the TRACKING loss (called every iteration of
    # FrontEnd.tracking()). Only the camera pose delta + exposure receive
    # gradients from it — the Gaussians are frozen during tracking.
    #
    # image_ab: rendered image after the per-frame affine exposure correction
    # exp(a) * I + b (see Camera.exposure_a/b). Correction is applied to the
    # RENDER rather than the GT so the map itself stays exposure-neutral.
    # `initialization` is unused here (kept for signature symmetry with
    # get_loss_mapping) and no caller passes it: the initialization frame
    # never goes through tracking — FrontEnd.initialize() assigns it the GT
    # pose and hands it straight to the backend.
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    # Photometric tracking loss (monocular case, and also the RGB term of the
    # RGB-D tracking loss). `depth` is unused.
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    # rgb_boundary_threshold: drop near-black GT pixels (sum of RGB below
    # threshold) — typically black borders from undistortion/rectification
    # that carry no real scene information.
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    # ...and additionally keep only high-gradient (edge) pixels, see
    # Camera.compute_grad_mask(): flat regions give almost no pose signal.
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    # Weighting by rendered `opacity` (per-pixel accumulated alpha) makes the
    # loss ignore pixels the map doesn't cover yet: there the rendering is
    # just background, and matching it would drag the pose toward nonsense.
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    # Mean over ALL pixels (masked ones contribute 0), so the loss magnitude
    # scales with the fraction of valid pixels.
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    # RGB-D tracking loss = alpha * photometric + (1 - alpha) * depth L1.
    # alpha (config Training.alpha, default 0.95) balances the two terms;
    # TUM RGB-D configs set 0.9 to lean more on depth.
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    # Depth term only where BOTH (a) the sensor returned a valid depth
    # (> 1 cm; 0 means "no reading") and (b) the map is well reconstructed
    # at that pixel (opacity > 0.95), otherwise the rendered depth is
    # meaningless and would bias the pose.
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    # Entry point for the MAPPING loss (BackEnd.initialize_map / map()).
    # Here gradients flow into the Gaussians AND into keyframe poses/exposure.
    #
    # initialization=True (BackEnd.initialize_map, first keyframe only): skip
    # the exposure correction. Numerically this is a no-op — exposure_a/b are
    # still 0 at that point, and exp(0)*I + 0 == I — it only avoids building
    # autograd edges to parameters no optimizer steps during initialization.
    # What actually makes frame 0 the exposure reference is BackEnd skipping
    # it (`current_window[cam_idx] == 0`) when building keyframe_optimizers,
    # so its exposure stays at identity forever.
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    # Photometric mapping loss (monocular). Differences vs. the tracking
    # version: NO edge (grad_mask) restriction and NO opacity weighting —
    # mapping must reconstruct every pixel, including flat regions and
    # not-yet-covered areas (that's precisely where new Gaussians need
    # gradients to grow/move into). `depth` is unused.
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    # RGB-D mapping loss = alpha * photometric + (1 - alpha) * depth L1.
    # Same idea as get_loss_mapping_rgb: no opacity mask on the depth term
    # (unlike tracking), because holes in the map are exactly what mapping
    # should fill using the sensor depth. `initialization` is unused.
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    # Robust "typical scene depth" of a RENDERED depth map, over pixels that
    # have positive depth, are well covered by the map (opacity > 0.95) and
    # pass the optional extra `mask`. Detached: pure statistic, no gradients.
    #
    # Used by the frontend for:
    #   - self.median_depth after tracking: makes keyframe thresholds
    #     (kf_translation, kf_min_translation) relative to scene scale, which
    #     is essential for monocular where the absolute scale is arbitrary;
    #   - add_new_keyframe (monocular): with return_std=True, to replace
    #     outlier depths by the median and pick the noise level used to
    #     initialize the new keyframe's Gaussians.
    # Returns median, or (median, std, valid_mask) if return_std.
    #
    # NOTE: despite the `opacity=None` default, opacity.detach() below runs
    # before the None check and would crash; every caller passes it.
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
