from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.nao import (
    NAO_ACTION_SCALE,
    NAO_CFG,
    NAO_FOOT_NAMES,
    NAO_JOINT_NAMES,
    NAO_TRACKING_BODY_NAMES,
)
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class NaoFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = NAO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Restrict the policy action to the 24 active joints; passive hand /
        # finger joints stay locked at their default pose.
        self.actions.joint_pos.joint_names = NAO_JOINT_NAMES
        self.actions.joint_pos.scale = NAO_ACTION_SCALE

        self.commands.motion.anchor_body_name = "torso"
        self.commands.motion.body_names = NAO_TRACKING_BODY_NAMES

        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso")

        # The default-pose randomizer indexes the action term's per-joint
        # `_offset` tensor (sized over the 24 action joints) using full
        # articulation joint indices. With NAO's 18 passive hand/finger joints
        # outside the action set, those indices go out of bounds. Disable the
        # offset randomization for NAO by passing pos_distribution_params=None
        # — the small (+-0.01 rad) calibration noise is not critical for the
        # tracking policy.
        self.events.add_joint_default_pos.params["pos_distribution_params"] = None

        wrist_body_names = ["l_wrist", "r_wrist"]
        self.rewards.motion_wrist_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 1.0, "body_names": wrist_body_names},
        )
        self.rewards.motion_wrist_ang_vel = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 3.14, "body_names": wrist_body_names},
        )

        # Penalize undesired contacts on every body except the feet and wrists.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!l_ankle$)(?!r_ankle$)(?!l_wrist$)(?!r_wrist$).+$"],
        )

        self.terminations.ee_body_pos.params["body_names"] = NAO_FOOT_NAMES
        # NAO is small (~0.58 m tall), so tighten anchor / EE thresholds from
        # the G1-sized defaults (0.25 m -> 0.15 m).
        self.terminations.anchor_pos.params["threshold"] = 0.15
        self.terminations.ee_body_pos.params["threshold"] = 0.15
