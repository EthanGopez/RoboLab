"""CuRobo v1 backend for the planner-neutral fixed-base manipulation API.

The installed Isaac Lab integration is pinned to CuRobo commit ebb7170. CuRobo
v2 intentionally stays behind this boundary until its Isaac Sim adapter lands.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch

from robolab.constants import PACKAGE_DIR
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    ARM_JOINT_NAMES,
    JointTrajectory,
    Pose as TaskPose,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    CAN_RADIUS,
    STATIC_COLLISION_BOXES,
)


_CONFIG_DIR = Path(PACKAGE_DIR) / "assets/robots/kinova_mobile_cobot/curobo"
_ROBOT_CONFIG = _CONFIG_DIR / "kinova_gen3_robolab.yml"
_ROBOT_URDF = _CONFIG_DIR / "kinova_gen3_robolab.urdf"
_SPHERES_CONFIG = _CONFIG_DIR / "kinova_gen3_robolab_spheres.yml"
_ARM_JOINT_LIMITS = (
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.24, 2.24),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.57, 2.57),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.09, 2.09),
    (-2.0 * math.pi, 2.0 * math.pi),
)
_START_LIMIT_EPSILON_RAD = 1.0e-4


def ensure_curobo_runtime_environment() -> None:
    """Use the Isaac environment's CUDA toolkit and an ABI-specific JIT cache.

    The CuRobo checkout contains extensions from an older PyTorch build. Its
    source fallback must therefore compile with the CUDA toolkit shipped in
    the active Isaac environment, not an older system ``nvcc``.
    """

    runtime_root = Path(sys.executable).resolve().parent.parent
    runtime_bin = runtime_root / "bin"
    if (runtime_bin / "nvcc").is_file():
        os.environ.setdefault("CUDA_HOME", str(runtime_root))
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if str(runtime_bin) not in path_parts:
            os.environ["PATH"] = os.pathsep.join([str(runtime_bin), *path_parts])

    torch_tag = torch.__version__.split("+")[0].replace(".", "_")
    cuda_tag = (torch.version.cuda or "cpu").replace(".", "_")
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    cache_dir = (
        Path(PACKAGE_DIR)
        / ".cache"
        / f"torch_extensions_curobo_{python_tag}_torch{torch_tag}_cuda{cuda_tag}"
    )
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_dir))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, ...]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quat_multiply(
        _quat_multiply(quaternion, (0.0, *vector)), _quat_conjugate(quaternion)
    )
    return (rotated[1], rotated[2], rotated[3])


def world_pose_to_base(
    pose_world: TaskPose,
    base_pose_xyzyaw: tuple[float, float, float, float],
) -> TaskPose:
    """Convert a task pose from kitchen coordinates to the fixed robot root."""

    bx, by, bz, yaw = base_pose_xyzyaw
    base_quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    inverse = _quat_conjugate(base_quat)
    delta = (
        pose_world.position[0] - bx,
        pose_world.position[1] - by,
        pose_world.position[2] - bz,
    )
    position = _quat_rotate(inverse, delta)
    quaternion = _quat_multiply(inverse, pose_world.quaternion_wxyz)
    return TaskPose(position, quaternion)


class CuroboFixedBasePlanner:
    """Collision-aware seven-joint planner with optional held-can spheres."""

    def __init__(
        self,
        base_pose_xyzyaw: tuple[float, float, float, float],
        *,
        cuda_device: int = 0,
        num_seeds: int = 8,
        collision_boxes=None,
    ) -> None:
        ensure_curobo_runtime_environment()
        # Lazy imports keep schema tests and non-CuRobo installations usable.
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import Cuboid, WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.motion_gen import (
            MotionGen,
            MotionGenConfig,
            MotionGenPlanConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CuRobo fixed-base planning requires a CUDA device")
        for path in (_ROBOT_CONFIG, _ROBOT_URDF, _SPHERES_CONFIG):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.base_pose = base_pose_xyzyaw
        self.tensor_args = TensorDeviceType(
            device=torch.device(f"cuda:{cuda_device}"), dtype=torch.float32
        )
        robot_cfg = load_yaml(str(_ROBOT_CONFIG))["robot_cfg"]
        robot_cfg["kinematics"]["urdf_path"] = str(_ROBOT_URDF)
        robot_cfg["kinematics"]["asset_root_path"] = str(_CONFIG_DIR)
        robot_cfg["kinematics"]["collision_spheres"] = str(_SPHERES_CONFIG)

        bx, by, bz, yaw = base_pose_xyzyaw
        inverse_yaw = -yaw
        world_cuboids = []
        if collision_boxes is None:
            collision_boxes = STATIC_COLLISION_BOXES
        for box in collision_boxes:
            dx = box.center_xyz[0] - bx
            dy = box.center_xyz[1] - by
            c, s = math.cos(inverse_yaw), math.sin(inverse_yaw)
            center_base = (
                c * dx - s * dy,
                s * dx + c * dy,
                box.center_xyz[2] - bz,
            )
            obstacle_yaw = getattr(box, "yaw", 0.0) - yaw
            world_cuboids.append(
                Cuboid(
                    name=f"kitchen_{box.name}",
                    dims=list(box.size_xyz),
                    pose=[
                        *center_base,
                        math.cos(obstacle_yaw / 2.0),
                        0.0,
                        0.0,
                        math.sin(obstacle_yaw / 2.0),
                    ],
                )
            )
        world_cfg = WorldConfig(cuboid=world_cuboids)
        motion_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args=self.tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            num_trajopt_seeds=num_seeds,
            num_graph_seeds=num_seeds,
            interpolation_dt=0.05,
            trajopt_tsteps=32,
            collision_activation_distance=0.010,
            position_threshold=0.008,
            rotation_threshold=0.08,
        )
        self.motion_gen = MotionGen(motion_cfg)
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=3,
            max_attempts=8,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5,
        )
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self._held_pose: TaskPose | None = None
        self.set_held_object(None)

    def set_held_object(
        self,
        pose_in_tool: TaskPose | None,
        *,
        object_spheres: tuple[tuple[float, float, float, float], ...] | None = None,
    ) -> None:
        """Represent a held object by conservative spheres on a virtual link."""

        capacity = 32
        spheres = torch.zeros(
            (capacity, 4), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        spheres[:, 3] = -10.0
        if pose_in_tool is not None:
            if object_spheres is None:
                object_spheres = tuple(
                    (0.0, 0.0, z_offset, CAN_RADIUS + 0.001)
                    for z_offset in (-0.026, 0.0, 0.026)
                )
            if len(object_spheres) > capacity:
                raise ValueError(f"held-object sphere count exceeds {capacity}")
            for index, (x_offset, y_offset, z_offset, radius) in enumerate(
                object_spheres
            ):
                offset = _quat_rotate(
                    pose_in_tool.quaternion_wxyz,
                    (x_offset, y_offset, z_offset),
                )
                spheres[index, :3] = torch.tensor(
                    [
                        pose_in_tool.position[0] + offset[0],
                        pose_in_tool.position[1] + offset[1],
                        pose_in_tool.position[2] + offset[2],
                    ],
                    device=self.tensor_args.device,
                )
                spheres[index, 3] = radius
        self.motion_gen.kinematics.kinematics_config.update_link_spheres(
            "attached_object", spheres
        )
        self._held_pose = pose_in_tool

    def plan_joint(
        self,
        start_joint_positions: tuple[float, ...],
        target_joint_positions: tuple[float, ...],
    ) -> JointTrajectory:
        """Plan a collision-checked path to a compact arm configuration."""

        from curobo.types.state import JointState

        if len(start_joint_positions) != len(ARM_JOINT_NAMES):
            raise ValueError("start_joint_positions must contain seven arm joints")
        if len(target_joint_positions) != len(ARM_JOINT_NAMES):
            raise ValueError("target_joint_positions must contain seven arm joints")
        names = list(ARM_JOINT_NAMES)
        sanitized_start = tuple(
            min(max(value, lower + _START_LIMIT_EPSILON_RAD), upper - _START_LIMIT_EPSILON_RAD)
            for value, (lower, upper) in zip(start_joint_positions, _ARM_JOINT_LIMITS)
        )
        start_state = JointState.from_position(
            torch.tensor(
                [sanitized_start],
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            ),
            joint_names=names,
        )
        goal_state = JointState.from_position(
            torch.tensor(
                [target_joint_positions],
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            ),
            joint_names=names,
        )
        started = time.perf_counter()
        result = self.motion_gen.plan_single_js(
            start_state, goal_state, self.plan_config
        )
        planning_time = time.perf_counter() - started
        if not bool(result.success.item()):
            raise RuntimeError(f"CuRobo joint planning failed: {result.status}")
        trajectory = result.get_interpolated_plan()
        if trajectory is None or len(trajectory.position) == 0:
            trajectory = result.optimized_plan
        if trajectory is None or len(trajectory.position) == 0:
            raise RuntimeError("CuRobo returned joint success without a trajectory")
        ordered = trajectory.get_ordered_joint_state(names)
        positions = tuple(
            tuple(float(value) for value in row)
            for row in ordered.position.detach().cpu().tolist()
        )
        return JointTrajectory(
            ARM_JOINT_NAMES, positions, planning_time, sample_dt_s=0.05
        )

    def plan(
        self,
        start_joint_positions: tuple[float, ...],
        target_pose_base: TaskPose,
        *,
        allow_hand_contact: bool = False,
    ) -> JointTrajectory:
        """Plan one segment and return only the seven commanded arm joints."""

        from curobo.types.math import Pose
        from curobo.types.state import JointState

        if len(start_joint_positions) != len(ARM_JOINT_NAMES):
            raise ValueError("start_joint_positions must contain seven arm joints")
        sanitized_start = tuple(
            min(max(value, lower + _START_LIMIT_EPSILON_RAD), upper - _START_LIMIT_EPSILON_RAD)
            for value, (lower, upper) in zip(start_joint_positions, _ARM_JOINT_LIMITS)
        )
        start = torch.tensor(
            [sanitized_start],
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        start_state = JointState.from_position(
            start, joint_names=list(ARM_JOINT_NAMES)
        )
        goal = Pose(
            position=torch.tensor(
                [target_pose_base.position],
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            ),
            quaternion=torch.tensor(
                [target_pose_base.quaternion_wxyz],
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            ),
        )

        kinematics = self.motion_gen.kinematics.kinematics_config
        if allow_hand_contact:
            kinematics.disable_link_spheres("tool_frame")
            kinematics.disable_link_spheres("robotiq_85_base_link")
            if self._held_pose is not None:
                kinematics.disable_link_spheres("attached_object")
        started = time.perf_counter()
        try:
            result = self.motion_gen.plan_single(start_state, goal, self.plan_config)
        finally:
            if allow_hand_contact:
                kinematics.enable_link_spheres("tool_frame")
                kinematics.enable_link_spheres("robotiq_85_base_link")
                if self._held_pose is not None:
                    kinematics.enable_link_spheres("attached_object")
        planning_time = time.perf_counter() - started
        if not bool(result.success.item()):
            raise RuntimeError(f"CuRobo planning failed: {result.status}")
        trajectory = result.get_interpolated_plan()
        if trajectory is None or len(trajectory.position) == 0:
            trajectory = result.optimized_plan
        if trajectory is None or len(trajectory.position) == 0:
            raise RuntimeError("CuRobo returned success without a trajectory")
        ordered = trajectory.get_ordered_joint_state(list(ARM_JOINT_NAMES))
        positions = tuple(
            tuple(float(value) for value in row)
            for row in ordered.position.detach().cpu().tolist()
        )
        return JointTrajectory(
            ARM_JOINT_NAMES, positions, planning_time, sample_dt_s=0.05
        )
