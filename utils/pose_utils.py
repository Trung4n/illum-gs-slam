import numpy as np
import torch

# Lie-algebra helpers for camera pose optimization.
#
# Tracking/mapping don't optimize R (3x3, must stay orthonormal) and T
# directly — instead each Camera carries a small 6-DoF tangent vector
# (cam_trans_delta, cam_rot_delta) that gradient descent is free to move
# anywhere in R^6, and update_pose() below "folds" it into the real pose via
# the SE(3) exponential map. This is the standard se(3)/SE(3) trick used to
# do gradient-based pose refinement without ever breaking the rotation's
# orthonormality constraint.
#
# Convention: a twist tau = [rho (translation part), theta (rotation part)],
# each in R^3. theta is an axis-angle vector: its direction is the rotation
# axis, its norm is the rotation angle in radians.


def rt2mat(R, T):
    # Pack a rotation matrix + translation vector into a 4x4 homogeneous
    # transform (numpy). Not called anywhere in this codebase currently —
    # kept as a small standalone utility.
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    # Build the skew-symmetric "hat" matrix [x]_x of a 3-vector x, such that
    # [x]_x @ v == cross(x, v) for any vector v. This is the building block
    # of the so(3) exponential map below (so(3) = skew-symmetric matrices).
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    # so(3) -> SO(3) exponential map (Rodrigues' rotation formula): turns an
    # axis-angle vector theta into a proper 3x3 rotation matrix.
    # Uses a first/second-order Taylor expansion near angle=0 to avoid the
    # sin(angle)/angle and (1-cos(angle))/angle^2 terms blowing up (0/0) for
    # very small rotations, which is the common case here since these are
    # per-iteration pose *deltas*, not full rotations.
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    # SO(3) "left Jacobian". For SE(3), the exponential map does NOT simply
    # copy the translation part rho of the twist into the output translation
    # (that would only be true for pure translations with no rotation) — it
    # must be pre-multiplied by V(theta) to account for the translation
    # sweeping along the rotation. See SE3_exp below. Same small-angle
    # Taylor-series safeguard as SO3_exp.
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    # se(3) -> SE(3) exponential map: turns a 6-vector twist tau = [rho, theta]
    # into a 4x4 homogeneous transform T (a small rigid-body motion). This is
    # what turns a raw, unconstrained gradient-descent step (tau, living in
    # flat R^6) into a valid rotation+translation that can be composed with
    # an existing camera pose.
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    # Applies one accumulated optimizer step to a camera's pose, and reports
    # whether tracking has converged. Called after every Adam.step() during
    # both frontend tracking (FrontEnd.tracking) and backend pose refinement
    # (BackEnd.map) — i.e. once per optimization iteration, per camera.
    #
    # tau: the 6-DoF pose delta for this step, as produced by gradient
    # descent on cam_trans_delta/cam_rot_delta (learnable nn.Parameters on
    # the Camera — see camera_utils.Camera).
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    # Current world-to-camera pose, as a 4x4 homogeneous matrix.
    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    # Left-multiply: apply the small delta on top of (i.e. "before", in
    # camera-then-world composition order) the current pose. This keeps the
    # result an exact rotation matrix, unlike naively doing R += dR.
    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    # A tiny step norm means the optimizer has essentially stopped moving
    # the pose -> tracking has converged; callers use this to break out of
    # their fixed-iteration-count optimization loop early.
    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    # The delta has now been absorbed into camera.R/T, so reset it to zero
    # before the next iteration accumulates a fresh one on top.
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
