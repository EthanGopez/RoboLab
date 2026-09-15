# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unassisted physical-contact controller for the validated XGrids2 can task."""

from __future__ import annotations

import numpy as np
import torch

from robolab.core.world.world_state import get_world
from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import JointTrajectory
from robolab.tasks.mobile_manipulation.kitchen_manipulation_predicates import (
    can_bilaterally_grasped,
)
from robolab.tasks.mobile_manipulation.xgrids2_grasp_audit import GraspAudit
from robolab.tasks.mobile_manipulation.xgrids2_lab_curobo_oracle import (
    XGRIDS2CuroboOracle,
    XGRIDS2CuroboResult,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    get_xgrids2_scenario,
)
from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import (
    placement_geometry,
)


class XGRIDS2UnassistedCanOracle(XGRIDS2CuroboOracle):
    """Use gripper contact alone; retention activation is a hard failure."""

    def __init__(self, env, *, num_seeds: int = 8, diagnostic_profile=None) -> None:
        super().__init__(
            env, num_seeds=num_seeds, diagnostic_profile=diagnostic_profile
        )
        source_z = float(self.can.data.root_pos_w[0, 2].item())
        self.grasp_audit = GraspAudit(float(env.step_dt), source_z)
        self._lift_paths: list[JointTrajectory] = []
        self._source_position = self.can.data.root_pos_w[0].detach().cpu().numpy().copy()
        self._previous_gripper_command: float | None = None
        self.unload_geometry = None
        # The generic retained-object controller releases from a deliberately
        # conservative height. The physical can grasp can lower closer to the
        # support before opening, avoiding an 8 mm free fall at placement.
        self.place_release_clearance_m = 0.034

    def _attach(self) -> None:
        raise RuntimeError("benchmark execution forbids fixed-joint retention")

    def _close_and_attach(self, result: XGRIDS2CuroboResult):
        self._source_position = self.can.data.root_pos_w[0].detach().cpu().numpy().copy()
        self.grasp_audit.source_center_z = float(self._source_position[2])
        result.phase_history.append("close_unassisted")
        self._stage_label = "close_unassisted"
        target = self._arm_target_positions()
        for _ in range(150):
            self._step(self._action(target, closed=True))
            if bool(can_bilaterally_grasped(self.env, stable_frames=4)[0].item()):
                measured = self.robot.data.joint_pos[0, self.gripper_drive_indices]
                self.gripper_close_command = (
                    measured.clone()
                    + torch.sign(self.gripper_close_command) * 0.025
                )
                self._hold(30, closed=True)
                if not bool(
                    can_bilaterally_grasped(self.env, stable_frames=4)[0].item()
                ):
                    raise RuntimeError("unassisted contact lost during preload")
                return self._held_pose()
        raise RuntimeError("unassisted bilateral contact was not established")

    def _plan_execute(self, planner, name, *args, **kwargs):
        result = args[-1]
        result.grasp_mode = "unassisted_physical_gripper_contact"
        result.release_mode = "support_verified_gradual_physical_opening"
        trajectory = super()._plan_execute(planner, name, *args, **kwargs)
        if name in {"lift", "lift_correction"}:
            self._lift_paths.append(trajectory)
        if name == "place_continuous":
            scenario = get_xgrids2_scenario("can_dining_to_snack")
            leg = scenario.legs[0]
            support = scenario.support(leg.place.support_name)
            position = (
                self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            ).detach().cpu().tolist()
            geometry = placement_geometry(
                scenario.object("can"), leg, support,
                {
                    "position_xyz": position,
                    "quaternion_wxyz": self.can.data.root_quat_w[0].detach().cpu().tolist(),
                },
            )
            clearance = geometry["bottom_z"] - support.surface_z
            if not geometry["contained"] or not -0.005 <= clearance <= 0.005:
                raise RuntimeError(f"cannot unload grasp away from support: {geometry}")
            result.phase_history.append("place_unload")
            self._stage_label = "place_unload"
            target = self._arm_target_positions()
            measured = self.robot.data.joint_pos[0, self.gripper_drive_indices].clone()
            for tick in range(30):
                self.gripper_close_command = (
                    measured
                    - torch.sign(measured) * 0.03 * ((tick + 1) / 30)
                )
                self._step(self._action(target, closed=True))
            self._hold(60, closed=True)
            self.unload_geometry = geometry
        return trajectory

    def _complete_stage(self, result, completed_stage):
        if completed_stage in {"pick", "carry"}:
            self._stage_label = "unassisted_hold_check"
            self._hold(90, closed=True)
            if not self.grasp_audit.lift_achieved or self.grasp_audit.drop_events:
                raise RuntimeError("sustained physical lift/hold gate failed")
        if completed_stage == "pick":
            result.phase_history.append("lower_to_source")
            self._stage_label = "lower_to_source"
            for path in reversed(self._lift_paths):
                self._execute(
                    JointTrajectory(
                        path.joint_names,
                        tuple(reversed(path.positions)),
                        0.0,
                        sample_dt_s=path.sample_dt_s,
                    ),
                    closed=True,
                )
            result.phase_history.append("release")
            self._stage_label = "release"
            self._hold(90, closed=False)
            position = self.can.data.root_pos_w[0].detach().cpu().numpy()
            if np.linalg.norm(position[:2] - self._source_position[:2]) > 0.04:
                raise RuntimeError("can did not return to source support")
            if float(torch.linalg.vector_norm(self.can.data.root_lin_vel_w[0])) > 0.05:
                raise RuntimeError("can did not settle after source release")
        if (
            self.grasp_audit.retention_ever
            or not self.grasp_audit.lift_achieved
            or self.grasp_audit.drop_events
        ):
            raise RuntimeError("unassisted grasp integrity gate failed")
        result.grasp_mode = "unassisted_physical_gripper_contact"
        result.release_mode = (
            "support_verified_gradual_physical_opening"
            if completed_stage in {"pick", "place", "full"}
            else "not_released_in_this_gate"
        )
        return super()._complete_stage(result, completed_stage)

    def _step(self, action):
        super()._step(action)
        retention = bool(self.grasp_joint.GetJointEnabledAttr().Get())
        if retention:
            raise RuntimeError("retention unexpectedly activated in benchmark mode")
        world = get_world(self.env)
        forces = [
            float(torch.linalg.vector_norm(world.get_contact_force("can", finger)[0]))
            for finger in ("left_fingertip", "right_fingertip")
        ]
        held = self._held_pose()
        command = float(action[0, self.slices["gripper"]].abs().max().item())
        release = (
            self.grasp_audit.released
            or (
                self.grasp_audit.lift_achieved
                and self._previous_gripper_command is not None
                and command < self._previous_gripper_command - 1.0e-5
            )
        )
        self._previous_gripper_command = command
        position = self.can.data.root_pos_w[0]
        self.grasp_audit.observe(
            center_z=float(position[2].item()),
            relative_xyz=held.position,
            retention=retention,
            left_force=forces[0],
            right_force=forces[1],
            release_commanded=release,
        )
        if self.grasp_audit.drop_events and not release:
            raise RuntimeError("object separated during unassisted transport")
