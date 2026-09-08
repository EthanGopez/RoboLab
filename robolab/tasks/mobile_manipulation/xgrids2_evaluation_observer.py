"""Isaac adapter for passive snapshots. No planner/phase-dependent scoring."""

from __future__ import annotations

import math

from robolab.tasks.mobile_manipulation.xgrids2_motion_diagnostics import ArmMotionDiagnostics


class PostStepObserver:
    """Scoped env.step observer shared by any policy using that environment.

    Calls the observer exactly once after a successfully returned environment
    step; never steps, resets, commands, or suppresses exceptions itself.
    """

    def __init__(self, env, callback):
        self.env, self.callback = env, callback

    def __enter__(self):
        self.original = self.env.step
        def step(action):
            result = self.original(action)
            self.callback(action, result)
            return result
        self.env.step = step
        return self

    def __exit__(self, *_):
        self.env.step = self.original


class IsaacTransferSampler:
    """Privileged evaluation-only state; never added to policy observations."""

    def __init__(self, env, scenario):
        from pxr import UsdPhysics
        from robolab.core.world.world_state import get_world

        self.env, self.scenario = env, scenario
        self.robot = env.scene["robot"]
        self.world = get_world(env)
        self.arm_names = [f"joint_{i}" for i in range(1, 8)]
        self.arm_indices = [self.robot.data.joint_names.index(n) for n in self.arm_names]
        self.slices = {}
        offset = 0
        for name, size in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
            self.slices[name] = slice(offset, offset+size)
            offset += size
        if self.slices["arm"].stop-self.slices["arm"].start != 7:
            raise ValueError("expected seven explicitly named arm joints")
        self.joints = {}
        for obj in scenario.objects:
            if obj.manipulated:
                prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[0]}/oracle_grasp_joint_{obj.name}")
                if scenario.name == "can_dining_to_snack":
                    from robolab.tasks.mobile_manipulation.xgrids2_lab_curobo_oracle import ORACLE_GRASP_JOINT_NAME
                    prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[0]}/{ORACLE_GRASP_JOINT_NAME}")
                if not prim.IsValid():
                    raise ValueError(f"missing retention instrumentation for {obj.name}")
                self.joints[obj.name] = UsdPhysics.FixedJoint(prim)
        self.motion = ArmMotionDiagnostics(float(env.step_dt), self._list(self.robot.data.joint_pos[0, self.arm_indices]))
        self.tick = 0
        self.flange_index = None
        if scenario.name == "can_dining_to_snack":
            from robolab.robots.kinova_mobile_cobot import MOBILE_KINOVA_FLANGE_BODY
            self.flange_index = self.robot.data.body_names.index(MOBILE_KINOVA_FLANGE_BODY)

    @staticmethod
    def _list(value):
        return value.detach().cpu().tolist()

    def sample(self, action):
        import torch

        self.tick += 1
        objects = {}
        for obj in self.scenario.objects:
            data = self.env.scene[obj.name].data
            objects[obj.name] = {
                "position_xyz": self._list(data.root_pos_w[0]-self.env.scene.env_origins[0]),
                "quaternion_wxyz": self._list(data.root_quat_w[0]),
                "linear_velocity_xyz": self._list(data.root_lin_vel_w[0]),
                "angular_velocity_xyz": self._list(data.root_ang_vel_w[0]),
                "retention_active": bool(self.joints[obj.name].GetJointEnabledAttr().Get()) if obj.name in self.joints else False,
                "fingertip_forces_n": [float(torch.linalg.vector_norm(self.world.get_contact_force(obj.name, finger)[0]).item())
                                       for finger in ("left_fingertip", "right_fingertip")],
            }
        q = self._list(self.robot.data.root_link_quat_w[0])
        w, x, y, z = q
        base = {"position_xyz": self._list(self.robot.data.root_link_pos_w[0]-self.env.scene.env_origins[0]),
                "yaw_rad": math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)),
                "linear_velocity_xyz": self._list(self.robot.data.root_lin_vel_w[0]),
                "angular_velocity_xyz": self._list(self.robot.data.root_ang_vel_w[0]),
                "command_velocity_body": self._list(action[0, self.slices["base"]])}
        motion = self.motion.update(target=self._list(action[0, self.slices["arm"]]),
                                    position=self._list(self.robot.data.joint_pos[0, self.arm_indices]),
                                    raw_velocity=self._list(self.robot.data.joint_vel[0, self.arm_indices]),
                                    retention_active=any(o["retention_active"] for o in objects.values()),
                                    phase="passive_environment_observation")
        if self.scenario.name == "can_dining_to_snack":
            term = self.env.action_manager.get_term("base")
            base.update({key: self._list(getattr(term, attr)[0]) for key, attr in (
                ("applied_velocity_body", "_applied_actions"),
                ("applied_acceleration_body", "_applied_acceleration"),
                ("applied_force_world_n", "_last_force_w"),
                ("applied_torque_world_nm", "_last_torque_w"))})
        snapshot = {"tick": self.tick, "base": base, "objects": objects, "motion": motion,
                    "arm_joint_names": self.arm_names}
        if self.flange_index is not None:
            index = self.flange_index
            snapshot["tool_flange"] = {
                "position_world_xyz": self._list(self.robot.data.body_pos_w[0, index]),
                "linear_velocity_world_xyz": self._list(self.robot.data.body_lin_vel_w[0, index]),
                "angular_velocity_world_xyz": self._list(self.robot.data.body_ang_vel_w[0, index]),
            }
        return snapshot
