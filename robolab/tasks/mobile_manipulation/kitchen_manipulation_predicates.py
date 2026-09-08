"""Strict vectorized predicates for the GenManip kitchen manipulation task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from isaaclab.utils import math as math_utils

from robolab.core.task.decorators import atomic
from robolab.core.world.world_state import get_world
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    BASE_POSITION_TOLERANCE,
    BASE_YAW_TOLERANCE_RAD,
    CAN_ANGULAR_SETTLE_SPEED,
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    CAN_LINEAR_SETTLE_SPEED,
    COUNTER_SURFACE_Z,
    GRASP_CONTACT_FORCE_N,
    GRASP_FORCE_NORMAL_CONE_RAD,
    GRASP_MAX_JOINT_TRACKING_ERROR_RAD,
    GRASP_MAX_PAD_SLIP_SPEED_M_S,
    GRASP_MAX_RELATIVE_POSITION_DRIFT_M,
    GRASP_MAX_RELATIVE_ROTATION_DRIFT_RAD,
    GRASP_NORMAL_OPPOSITION_TOLERANCE_RAD,
    GRASP_SECURE_FORCE_N,
    GRASP_SECURE_STABLE_FRAMES,
    GRASP_STABLE_FRAMES,
    ISLAND_SURFACE_Z,
    MIN_LIFT_CLEARANCE,
    PICK_REGION,
    PLACEMENT_STABLE_FRAMES,
    PLACE_REGION,
    STAGING_POSES,
    base_pose_is_clear,
)


def _select(value, env_id: int | None):
    if env_id is None:
        return value
    if isinstance(value, torch.Tensor):
        selected = value[env_id]
        return bool(selected.item()) if selected.numel() == 1 else selected
    return value


def _stable_boolean(
    env,
    raw: torch.Tensor,
    *,
    key: str,
    frames: int,
    env_id: int | None,
):
    """Require a per-environment boolean to remain true for consecutive frames."""

    world = get_world(env)

    def factory():
        count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        def reset(ids):
            count[ids] = 0

        return {"count": count, "__reset__": reset}

    state = world.get_or_init_predicate_state(key, factory)
    count = state["count"]
    if env_id is None:
        count[:] = torch.where(raw, count + 1, torch.zeros_like(count))
        return count >= frames
    count[env_id] = count[env_id] + 1 if bool(raw[env_id].item()) else 0
    return bool((count[env_id] >= frames).item())


def _can_state(env):
    can = env.scene["can"]
    position = can.data.root_pos_w - env.scene.env_origins
    linear_speed = torch.linalg.vector_norm(can.data.root_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(can.data.root_ang_vel_w, dim=1)
    return position, linear_speed, angular_speed


def _inside_region(position: torch.Tensor, region, tolerance: float = 0.0) -> torch.Tensor:
    return (
        (position[:, 0] >= region.xmin - tolerance)
        & (position[:, 0] <= region.xmax + tolerance)
        & (position[:, 1] >= region.ymin - tolerance)
        & (position[:, 1] <= region.ymax + tolerance)
    )


def _instant_bilateral_grasp(
    env, min_force_n: float, *, object_name: str = "can"
) -> torch.Tensor:
    world = get_world(env)
    left = world.in_contact(object_name, "left_fingertip", force_threshold=min_force_n)
    right = world.in_contact(object_name, "right_fingertip", force_threshold=min_force_n)
    left_force = torch.linalg.vector_norm(
        world.get_contact_force(object_name, "left_fingertip"), dim=-1
    )
    right_force = torch.linalg.vector_norm(
        world.get_contact_force(object_name, "right_fingertip"), dim=-1
    )
    return (
        left
        & right
        & (left_force >= min_force_n)
        & (right_force >= min_force_n)
    )



@dataclass
class PhysicalGraspQuality:
    """Vectorized measurements used to qualify an unassisted pinch grasp."""

    left_force_w: torch.Tensor
    right_force_w: torch.Tensor
    left_normal_force: torch.Tensor
    right_normal_force: torch.Tensor
    left_force_normal_cosine: torch.Tensor
    right_force_normal_cosine: torch.Tensor
    normal_opposition_cosine: torch.Tensor
    left_slip_speed: torch.Tensor
    right_slip_speed: torch.Tensor
    left_raw_slip_speed: torch.Tensor
    right_raw_slip_speed: torch.Tensor
    max_joint_tracking_error: torch.Tensor
    max_gripper_applied_effort: torch.Tensor
    relative_position_drift: torch.Tensor
    relative_rotation_drift: torch.Tensor
    bilateral_contact: torch.Tensor
    instantaneous_valid: torch.Tensor
    stable_frames: torch.Tensor
    secure: torch.Tensor

    def env_dict(self, env_id: int) -> dict:
        """Convert one environment's measurements to JSON-safe diagnostics."""

        vector_names = ("left_force_w", "right_force_w")
        result = {
            name: getattr(self, name)[env_id].detach().cpu().tolist()
            for name in vector_names
        }
        scalar_names = (
            "left_normal_force",
            "right_normal_force",
            "left_force_normal_cosine",
            "right_force_normal_cosine",
            "normal_opposition_cosine",
            "left_slip_speed",
            "right_slip_speed",
            "left_raw_slip_speed",
            "right_raw_slip_speed",
            "max_joint_tracking_error",
            "max_gripper_applied_effort",
            "relative_position_drift",
            "relative_rotation_drift",
        )
        result.update(
            {name: float(getattr(self, name)[env_id].item()) for name in scalar_names}
        )
        result.update(
            {
                "bilateral_contact": bool(self.bilateral_contact[env_id].item()),
                "instantaneous_valid": bool(self.instantaneous_valid[env_id].item()),
                "stable_frames": int(self.stable_frames[env_id].item()),
                "secure": bool(self.secure[env_id].item()),
            }
        )
        return result


def _pad_contact_kinematics(
    robot,
    can,
    *,
    body_name: str,
    pad_offset: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pad-to-can normal and pad-relative contact-point position."""

    body_index = robot.data.body_names.index(body_name)
    body_pos = robot.data.body_pos_w[:, body_index]
    body_quat = robot.data.body_quat_w[:, body_index]
    offset = torch.tensor(
        pad_offset, dtype=body_pos.dtype, device=body_pos.device
    ).repeat(body_pos.shape[0], 1)
    pad_pos = body_pos + math_utils.quat_apply(body_quat, offset)
    inward_normal = can.data.root_pos_w - pad_pos
    inward_normal = inward_normal / torch.clamp(
        torch.linalg.vector_norm(inward_normal, dim=1, keepdim=True), min=1.0e-6
    )

    contact_point = can.data.root_pos_w - CAN_RADIUS * inward_normal
    return inward_normal, contact_point - pad_pos


class PhysicalGraspMonitor:
    """Five-part bilateral force, pose-stability, and slip predicate."""

    def __init__(
        self,
        env,
        *,
        flange_body_name: str,
        left_fingertip_body_name: str,
        right_fingertip_body_name: str,
        fingertip_pad_offsets: dict[str, tuple[float, float, float]],
        gripper_joint_names: tuple[str, ...],
    ):
        """Bind generic grasp-quality logic to one robot embodiment."""

        self.env = env
        self.left_fingertip_body_name = left_fingertip_body_name
        self.right_fingertip_body_name = right_fingertip_body_name
        self.fingertip_pad_offsets = fingertip_pad_offsets
        self.robot = env.scene["robot"]
        self.can = env.scene["can"]
        self.flange_index = self.robot.data.body_names.index(flange_body_name)
        self.gripper_joint_indices = [
            self.robot.data.joint_names.index(name) for name in gripper_joint_names
        ]
        self.reference_pos_f = torch.zeros((env.num_envs, 3), device=env.device)
        self.reference_quat_f = torch.zeros((env.num_envs, 4), device=env.device)
        self.reference_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.stable_frames = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.left_filtered_slip_velocity = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self.right_filtered_slip_velocity = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self.left_previous_contact_vector = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self.right_previous_contact_vector = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self.slip_reference_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.control_dt = float(env.step_dt)
        self.last: PhysicalGraspQuality | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.reference_pos_f[env_ids] = 0.0
        self.reference_quat_f[env_ids] = 0.0
        self.reference_valid[env_ids] = False
        self.stable_frames[env_ids] = 0
        self.left_filtered_slip_velocity[env_ids] = 0.0
        self.right_filtered_slip_velocity[env_ids] = 0.0
        self.left_previous_contact_vector[env_ids] = 0.0
        self.right_previous_contact_vector[env_ids] = 0.0
        self.slip_reference_valid[env_ids] = False

    def update(self, joint_targets: torch.Tensor) -> PhysicalGraspQuality:
        """Measure the grasp and update its consecutive-frame history."""

        world = get_world(self.env)
        left_force = world.get_contact_force("can", "left_fingertip")
        right_force = world.get_contact_force("can", "right_fingertip")
        left_magnitude = torch.linalg.vector_norm(left_force, dim=1)
        right_magnitude = torch.linalg.vector_norm(right_force, dim=1)
        left_normal, left_contact_vector = _pad_contact_kinematics(
            self.robot,
            self.can,
            body_name=self.left_fingertip_body_name,
            pad_offset=self.fingertip_pad_offsets[self.left_fingertip_body_name],
        )
        right_normal, right_contact_vector = _pad_contact_kinematics(
            self.robot,
            self.can,
            body_name=self.right_fingertip_body_name,
            pad_offset=self.fingertip_pad_offsets[self.right_fingertip_body_name],
        )
        # Estimate slip from the measured change in the can contact point
        # relative to each pad. Isaac Lab articulation velocity buffers can
        # expose pre-constraint velocities during hard contact, so directly
        # subtracting those buffers reports apparent slip even while the
        # constrained poses are stationary. Pose differencing measures the
        # post-solver motion that determines whether the can is actually
        # migrating across a pad.
        slip_reference_valid = self.slip_reference_valid
        left_relative_velocity = (
            left_contact_vector - self.left_previous_contact_vector
        ) / self.control_dt
        right_relative_velocity = (
            right_contact_vector - self.right_previous_contact_vector
        ) / self.control_dt
        left_relative_velocity = torch.where(
            slip_reference_valid[:, None],
            left_relative_velocity,
            torch.zeros_like(left_relative_velocity),
        )
        right_relative_velocity = torch.where(
            slip_reference_valid[:, None],
            right_relative_velocity,
            torch.zeros_like(right_relative_velocity),
        )
        left_tangential_velocity = left_relative_velocity - (
            torch.sum(left_relative_velocity * left_normal, dim=1, keepdim=True)
            * left_normal
        )
        right_tangential_velocity = right_relative_velocity - (
            torch.sum(right_relative_velocity * right_normal, dim=1, keepdim=True)
            * right_normal
        )
        slip_alpha = 0.2
        self.left_filtered_slip_velocity.mul_(1.0 - slip_alpha).add_(
            left_tangential_velocity, alpha=slip_alpha
        )
        self.right_filtered_slip_velocity.mul_(1.0 - slip_alpha).add_(
            right_tangential_velocity, alpha=slip_alpha
        )
        self.left_previous_contact_vector[:] = left_contact_vector
        self.right_previous_contact_vector[:] = right_contact_vector
        self.slip_reference_valid[:] = True
        left_slip = torch.linalg.vector_norm(
            self.left_filtered_slip_velocity, dim=1
        )
        right_slip = torch.linalg.vector_norm(
            self.right_filtered_slip_velocity, dim=1
        )
        left_raw_slip = torch.linalg.vector_norm(left_tangential_velocity, dim=1)
        right_raw_slip = torch.linalg.vector_norm(right_tangential_velocity, dim=1)

        left_normal_force = torch.sum(left_force * left_normal, dim=1)
        right_normal_force = torch.sum(right_force * right_normal, dim=1)
        left_alignment = left_normal_force / torch.clamp(left_magnitude, min=1.0e-6)
        right_alignment = right_normal_force / torch.clamp(
            right_magnitude, min=1.0e-6
        )
        left_force_direction = left_force / torch.clamp(
            left_magnitude[:, None], min=1.0e-6
        )
        right_force_direction = right_force / torch.clamp(
            right_magnitude[:, None], min=1.0e-6
        )
        opposition = -torch.sum(
            left_force_direction * right_force_direction, dim=1
        )
        bilateral = (
            world.in_contact("can", "left_fingertip", force_threshold=0.1)
            & world.in_contact("can", "right_fingertip", force_threshold=0.1)
        )
        actual = self.robot.data.joint_pos[:, self.gripper_joint_indices]
        joint_error = torch.max(torch.abs(joint_targets - actual), dim=1).values
        applied_effort = torch.max(
            torch.abs(self.robot.data.applied_torque[:, self.gripper_joint_indices]),
            dim=1,
        ).values
        candidate = (
            bilateral
            & (left_normal_force >= GRASP_SECURE_FORCE_N)
            & (right_normal_force >= GRASP_SECURE_FORCE_N)
            & (left_alignment >= math.cos(GRASP_FORCE_NORMAL_CONE_RAD))
            & (right_alignment >= math.cos(GRASP_FORCE_NORMAL_CONE_RAD))
            & (opposition >= math.cos(GRASP_NORMAL_OPPOSITION_TOLERANCE_RAD))
            & (left_slip <= GRASP_MAX_PAD_SLIP_SPEED_M_S)
            & (right_slip <= GRASP_MAX_PAD_SLIP_SPEED_M_S)
            & (joint_error <= GRASP_MAX_JOINT_TRACKING_ERROR_RAD)
        )

        flange_pos = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quat = self.robot.data.body_quat_w[:, self.flange_index]
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(
            flange_pos,
            flange_quat,
            self.can.data.root_pos_w,
            self.can.data.root_quat_w,
        )
        new_reference = candidate & ~self.reference_valid
        self.reference_pos_f[new_reference] = relative_pos[new_reference]
        self.reference_quat_f[new_reference] = relative_quat[new_reference]
        self.reference_valid[new_reference] = True
        position_drift = torch.linalg.vector_norm(
            relative_pos - self.reference_pos_f, dim=1
        )
        rotation_drift = math_utils.quat_error_magnitude(
            relative_quat, self.reference_quat_f
        )
        instantaneous_valid = (
            candidate
            & self.reference_valid
            & (position_drift <= GRASP_MAX_RELATIVE_POSITION_DRIFT_M)
            & (rotation_drift <= GRASP_MAX_RELATIVE_ROTATION_DRIFT_RAD)
        )
        self.stable_frames[:] = torch.where(
            instantaneous_valid,
            self.stable_frames + 1,
            torch.zeros_like(self.stable_frames),
        )
        self.reference_valid[~instantaneous_valid] = False
        secure = self.stable_frames >= GRASP_SECURE_STABLE_FRAMES
        self.last = PhysicalGraspQuality(
            left_force_w=left_force,
            right_force_w=right_force,
            left_normal_force=left_normal_force,
            right_normal_force=right_normal_force,
            left_force_normal_cosine=left_alignment,
            right_force_normal_cosine=right_alignment,
            normal_opposition_cosine=opposition,
            left_slip_speed=left_slip,
            right_slip_speed=right_slip,
            left_raw_slip_speed=left_raw_slip,
            right_raw_slip_speed=right_raw_slip,
            max_joint_tracking_error=joint_error,
            max_gripper_applied_effort=applied_effort,
            relative_position_drift=position_drift,
            relative_rotation_drift=rotation_drift,
            bilateral_contact=bilateral,
            instantaneous_valid=instantaneous_valid,
            stable_frames=self.stable_frames.clone(),
            secure=secure,
        )
        return self.last
@atomic
def can_supported_by_island(
    env,
    env_id: int | None = None,
    *,
    region=PICK_REGION,
    support_name: str = "island_top_east",
):
    world = get_world(env)
    position, linear_speed, angular_speed = _can_state(env)
    force_supported = world.is_supported_on_surface("can", support_name)
    expected_z = ISLAND_SURFACE_Z + CAN_HALF_HEIGHT
    in_region = _inside_region(position, region, tolerance=0.005)
    at_surface = torch.abs(position[:, 2] - expected_z) <= 0.006
    settled = (
        (linear_speed < CAN_LINEAR_SETTLE_SPEED)
        & (angular_speed < CAN_ANGULAR_SETTLE_SPEED)
    )
    # GPU PhysX can report zero filtered force after an object becomes fully
    # quiescent. The position/velocity fallback is intentionally conjunctive:
    # it accepts only a settled body at the audited support height and bounds.
    result = in_region & at_surface & settled & (force_supported | settled)
    return _select(result, env_id)



@atomic
def object_bilaterally_grasped(
    env,
    object_name: str,
    min_force_n: float = GRASP_CONTACT_FORCE_N,
    stable_frames: int = GRASP_STABLE_FRAMES,
    env_id: int | None = None,
):
    raw = _instant_bilateral_grasp(env, min_force_n, object_name=object_name)
    result = _stable_boolean(
        env,
        raw,
        key=f"kitchen_bilateral_grasp:{object_name}:{min_force_n}:{stable_frames}",
        frames=stable_frames,
        env_id=env_id,
    )
    return result



@atomic
def can_bilaterally_grasped(
    env,
    min_force_n: float = GRASP_CONTACT_FORCE_N,
    stable_frames: int = GRASP_STABLE_FRAMES,
    env_id: int | None = None,
):
    return object_bilaterally_grasped(
        env,
        "can",
        min_force_n=min_force_n,
        stable_frames=stable_frames,
        env_id=env_id,
    )


@atomic
def can_lifted_clear(
    env,
    clearance: float = MIN_LIFT_CLEARANCE,
    env_id: int | None = None,
    *,
    support_name: str = "island_top_east",
):
    world = get_world(env)
    position, _, _ = _can_state(env)
    can_bottom = position[:, 2] - CAN_HALF_HEIGHT
    island_contact = world.in_contact("can", support_name, force_threshold=0.1)
    result = (
        _instant_bilateral_grasp(env, GRASP_CONTACT_FORCE_N)
        & (can_bottom >= ISLAND_SURFACE_Z + clearance)
        & ~island_contact
    )
    return _select(result, env_id)


@atomic
def base_at_place_stage(env, env_id: int | None = None):
    robot = env.scene["robot"]
    position = robot.data.root_link_pos_w - env.scene.env_origins
    quaternion = robot.data.root_link_quat_w
    target = STAGING_POSES["place"]
    xy_error = torch.linalg.vector_norm(
        position[:, :2] - torch.tensor(target[:2], device=env.device), dim=1
    )
    w, x, y, z = quaternion.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_error = torch.remainder(yaw - target[3] + math.pi, 2.0 * math.pi) - math.pi
    linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    angular_speed = torch.abs(robot.data.root_link_ang_vel_w[:, 2])
    result = (
        (xy_error <= BASE_POSITION_TOLERANCE)
        & (torch.abs(yaw_error) <= BASE_YAW_TOLERANCE_RAD)
        & (linear_speed < 0.02)
        & (angular_speed < 0.03)
    )
    return _select(result, env_id)


@atomic
def gripper_released(
    env,
    stable_frames: int = 5,
    env_id: int | None = None,
):
    world = get_world(env)
    left = world.in_contact("can", "left_fingertip", force_threshold=0.1)
    right = world.in_contact("can", "right_fingertip", force_threshold=0.1)
    # Release is an object-level semantic: neither contact patch still holds
    # the can. Whether a particular hand is mechanically "open" belongs to
    # that embodiment's controller, not the task success predicate.
    raw = (~left) & (~right)
    return _stable_boolean(
        env,
        raw,
        key=f"kitchen_gripper_released:{stable_frames}",
        frames=stable_frames,
        env_id=env_id,
    )


@atomic
def can_supported_in_place_region(
    env,
    require_released: bool = True,
    stable_frames: int = PLACEMENT_STABLE_FRAMES,
    env_id: int | None = None,
):
    world = get_world(env)
    position, linear_speed, angular_speed = _can_state(env)
    force_supported = world.is_supported_on_surface("can", "counter_right_top")
    expected_z = COUNTER_SURFACE_Z + CAN_HALF_HEIGHT
    in_region = _inside_region(position, PLACE_REGION, tolerance=0.005)
    at_surface = torch.abs(position[:, 2] - expected_z) <= 0.006
    settled = (
        (linear_speed < CAN_LINEAR_SETTLE_SPEED)
        & (angular_speed < CAN_ANGULAR_SETTLE_SPEED)
    )
    raw = in_region & at_surface & settled & (force_supported | settled)
    if require_released:
        left = world.in_contact("can", "left_fingertip", force_threshold=0.1)
        right = world.in_contact("can", "right_fingertip", force_threshold=0.1)
        raw &= ~left & ~right
    return _stable_boolean(
        env,
        raw,
        key=f"kitchen_place_stable:{require_released}:{stable_frames}",
        frames=stable_frames,
        env_id=env_id,
    )


@atomic
def robot_collision_free(env, env_id: int | None = None):
    robot = env.scene["robot"]
    can = env.scene["can"]
    robot_finite = torch.isfinite(robot.data.root_link_state_w).all(dim=1)
    can_finite = torch.isfinite(can.data.root_state_w).all(dim=1)
    can_position = can.data.root_pos_w - env.scene.env_origins
    can_in_workspace = (
        (can_position[:, 0] > -1.8)
        & (can_position[:, 0] < 3.8)
        & (can_position[:, 1] > -2.8)
        & (can_position[:, 1] < 2.5)
        & (can_position[:, 2] > -0.1)
        & (can_position[:, 2] < 3.0)
    )
    position = robot.data.root_link_pos_w - env.scene.env_origins
    geometric_clearance = torch.tensor(
        [base_pose_is_clear((float(row[0]), float(row[1]))) for row in position],
        dtype=torch.bool,
        device=env.device,
    )
    forbidden = getattr(env, "_kitchen_forbidden_collision", None)
    if forbidden is None:
        forbidden = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _select(
        robot_finite
        & can_finite
        & can_in_workspace
        & geometric_clearance
        & ~forbidden,
        env_id,
    )


@atomic
def kitchen_transfer_success(env, env_id: int | None = None):
    """Ordered success: grasp, lift, arrive while held, then stable released place."""

    world = get_world(env)

    def factory():
        grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        lifted = torch.zeros_like(grasped)
        arrived_held = torch.zeros_like(grasped)

        def reset(ids):
            grasped[ids] = False
            lifted[ids] = False
            arrived_held[ids] = False

        return {
            "grasped": grasped,
            "lifted": lifted,
            "arrived_held": arrived_held,
            "__reset__": reset,
        }

    state = world.get_or_init_predicate_state("kitchen_transfer_order", factory)
    bilateral = _instant_bilateral_grasp(env, GRASP_CONTACT_FORCE_N)
    lifted_now = can_lifted_clear(env)
    arrived_now = base_at_place_stage(env) & bilateral
    state["grasped"] |= bilateral
    state["lifted"] |= state["grasped"] & lifted_now
    state["arrived_held"] |= state["lifted"] & arrived_now
    placed = can_supported_in_place_region(env, require_released=True)
    released = gripper_released(env)
    collision_free = robot_collision_free(env)
    result = (
        state["grasped"]
        & state["lifted"]
        & state["arrived_held"]
        & placed
        & released
        & collision_free
    )
    return _select(result, env_id)
