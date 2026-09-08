"""Mobile Kinova environment registrations."""

from .auto_env_registrations_jointpos import (
    ENV_NAME,
    register_genmanip_kitchen_mobile_env,
)
from .auto_env_registrations_ik import (
    IK_ENV_NAME,
    register_genmanip_kitchen_mobile_ik_env,
)
from .auto_env_registrations_xgrids2_lab import (
    XGRIDS2_LAB_ENV_NAME,
    register_xgrids2_lab_mobile_env,
)
from .auto_env_registrations_xgrids2_lab_manipulation import (
    XGRIDS2_LAB_MANIP_ENV_NAME,
    XGRIDS2_LAB_MANIP_IK_ENV_NAME,
    register_xgrids2_lab_manipulation_env,
    register_xgrids2_lab_manipulation_ik_env,
)
from .auto_env_registrations_xgrids2_lab_plate_sink import (
    XGRIDS2_LAB_PLATE_SINK_ENV_NAME,
    register_xgrids2_lab_plate_sink_env,
)
from .auto_env_registrations_xgrids2_scenarios import register_xgrids2_scenario

__all__ = [
    "ENV_NAME",
    "IK_ENV_NAME",
    "register_genmanip_kitchen_mobile_env",
    "register_genmanip_kitchen_mobile_ik_env",
    "XGRIDS2_LAB_ENV_NAME",
    "register_xgrids2_lab_mobile_env",
    "XGRIDS2_LAB_MANIP_ENV_NAME",
    "XGRIDS2_LAB_MANIP_IK_ENV_NAME",
    "register_xgrids2_lab_manipulation_env",
    "register_xgrids2_lab_manipulation_ik_env",
    "XGRIDS2_LAB_PLATE_SINK_ENV_NAME",
    "register_xgrids2_lab_plate_sink_env",
    "register_xgrids2_scenario",
]
