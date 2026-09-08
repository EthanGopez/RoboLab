# SPDX-License-Identifier: Apache-2.0

"""Small, simulator-independent helpers for holonomic mobile-base control."""

import torch


def body_twist_to_world(body_twist: torch.Tensor, root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert ``[vx, vy, wz]`` body commands to six-dimensional world twists."""
    w, x, y, z = root_quat_wxyz.unbind(dim=-1)
    yaw_sin = 2.0 * (w * z + x * y)
    yaw_cos = 1.0 - 2.0 * (y * y + z * z)
    vx, vy, wz = body_twist.unbind(dim=-1)
    world = torch.zeros((*body_twist.shape[:-1], 6), device=body_twist.device, dtype=body_twist.dtype)
    world[..., 0] = yaw_cos * vx - yaw_sin * vy
    world[..., 1] = yaw_sin * vx + yaw_cos * vy
    world[..., 5] = wz
    return world


def slew_rate_limit(current: torch.Tensor, target: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
    """Move each command component toward its target by at most ``max_delta``."""
    return current + torch.clamp(target - current, min=-max_delta, max=max_delta)


def jerk_limited_velocity_step(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    max_acceleration: torch.Tensor,
    max_jerk: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound actual finite-difference acceleration AND jerk, including arrival.

    A damped second-order velocity servo decelerates acceleration before arrival.
    Do not clamp velocity to the target or reset acceleration there: either
    operation violates the advertised finite-difference jerk bound. Settling is
    asymptotic and large saturated reversals may overshoot; pose feedback remains
    responsible for goal convergence. Reset is reserved for episode boundaries.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    frequency = max_jerk / torch.clamp(max_acceleration, min=1e-9)
    desired_jerk = frequency**2 * (target_velocity - velocity) - 2*frequency*acceleration
    next_acceleration = torch.clamp(
        acceleration + torch.clamp(desired_jerk, min=-max_jerk, max=max_jerk)*dt,
        min=-max_acceleration, max=max_acceleration,
    )
    next_velocity = velocity + next_acceleration * dt
    return next_velocity, next_acceleration


def clamp_vector_norm(vectors: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    """Clamp vectors by magnitude without changing their directions."""

    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scale = torch.clamp(maximum_norm / torch.clamp(norms, min=1.0e-9), max=1.0)
    return vectors * scale


def planar_base_wrench(
    *,
    target_world_twist: torch.Tensor,
    measured_world_twist: torch.Tensor,
    root_height: torch.Tensor,
    target_height: torch.Tensor,
    body_z_axis_world: torch.Tensor,
    linear_velocity_gain: float,
    angular_velocity_gain: float,
    height_stiffness: float,
    height_damping: float,
    tilt_stiffness: float,
    tilt_damping: float,
    maximum_planar_force: float,
    maximum_vertical_force: float,
    maximum_yaw_torque: float,
    maximum_tilt_torque: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a bounded world-frame wrench for an ideal dynamic planar base.

    The planar axes are velocity controlled. Height, roll, and pitch are
    finite-gain constraints, so the base remains part of the rigid-body solve
    instead of having its state overwritten after every physics tick.
    """

    force = torch.zeros_like(measured_world_twist[..., :3])
    torque = torch.zeros_like(force)

    planar_error = target_world_twist[..., :2] - measured_world_twist[..., :2]
    force[..., :2] = clamp_vector_norm(
        linear_velocity_gain * planar_error, maximum_planar_force
    )
    force[..., 2] = torch.clamp(
        height_stiffness * (target_height - root_height)
        - height_damping * measured_world_twist[..., 2],
        min=-maximum_vertical_force,
        max=maximum_vertical_force,
    )

    world_up = torch.zeros_like(body_z_axis_world)
    world_up[..., 2] = 1.0
    tilt_axis_error = torch.linalg.cross(body_z_axis_world, world_up, dim=-1)
    torque[..., :2] = clamp_vector_norm(
        tilt_stiffness * tilt_axis_error[..., :2]
        - tilt_damping * measured_world_twist[..., 3:5],
        maximum_tilt_torque,
    )
    torque[..., 2] = torch.clamp(
        angular_velocity_gain
        * (target_world_twist[..., 5] - measured_world_twist[..., 5]),
        min=-maximum_yaw_torque,
        max=maximum_yaw_torque,
    )
    return force, torque


def quaternion_z_axis_world(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return each quaternion's local positive-Z axis in world coordinates."""

    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    return torch.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )
