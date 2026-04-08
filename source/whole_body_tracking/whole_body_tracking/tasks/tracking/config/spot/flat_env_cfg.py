from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.spot import SPOT_ACTION_SCALE, SPOT_CFG, SPOT_FOOT_NAMES, SPOT_TRACKING_BODY_NAMES
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class SpotFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = SPOT_ACTION_SCALE
        self.commands.motion.anchor_body_name = "spot"
        self.commands.motion.body_names = SPOT_TRACKING_BODY_NAMES

        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="spot")
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!fl_foot$)(?!fr_foot$)(?!hl_foot$)(?!hr_foot$).+$"],
        )
        self.terminations.ee_body_pos.params["body_names"] = SPOT_FOOT_NAMES
