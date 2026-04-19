import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

NAO_URDF_PATH = f"{ASSET_DIR}/nao/nao.urdf"

# Active 24 joints, in the same order as the GMR-retargeted pkl `dof_pos` columns
# (which follows the MJCF qpos order in `GMR/assets/nao/nao_scene.xml` after the free joint).
NAO_JOINT_NAMES = [
    "HeadYaw",
    "HeadPitch",
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    "RHipYawPitch",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
]

# Full 40-column pkl `dof_pos` order (active 24 + 16 finger columns that are
# constant zero in GMR retargets). Used by `pkl_to_npz.py` to map columns to
# articulation joints by name regardless of URDF traversal order.
NAO_PKL_DOF_ORDER = [
    "HeadYaw",
    "HeadPitch",
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    "RHipYawPitch",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LFinger21",
    "LFinger22",
    "LFinger23",
    "LFinger11",
    "LFinger12",
    "LFinger13",
    "LThumb1",
    "LThumb2",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "RFinger21",
    "RFinger22",
    "RFinger23",
    "RFinger11",
    "RFinger12",
    "RFinger13",
    "RThumb1",
    "RThumb2",
]

# Passive joints in the URDF that are NOT in the pkl active set — kept stiffly
# locked at zero to avoid extraneous degrees of freedom in the policy.
NAO_PASSIVE_JOINT_NAMES = [
    "LHand",
    "RHand",
    "LFinger11", "LFinger12", "LFinger13",
    "LFinger21", "LFinger22", "LFinger23",
    "LThumb1", "LThumb2",
    "RFinger11", "RFinger12", "RFinger13",
    "RFinger21", "RFinger22", "RFinger23",
    "RThumb1", "RThumb2",
]

# Bodies tracked by the motion command (anchor + key kinematic-chain links).
NAO_TRACKING_BODY_NAMES = [
    "torso",
    "Head",
    "LPelvis",
    "LThigh",
    "LTibia",
    "l_ankle",
    "RPelvis",
    "RThigh",
    "RTibia",
    "r_ankle",
    "LBicep",
    "LForeArm",
    "l_wrist",
    "RBicep",
    "RForeArm",
    "r_wrist",
]

# End-effector / contact-allowed link names (feet + wrists for grounding).
NAO_FOOT_NAMES = ["l_ankle", "r_ankle", "l_wrist", "r_wrist"]

# A modest standing pose with slight knee bend so the robot does not fall on
# the very first frame before the motion command takes over.
NAO_STANDING_JOINT_POS = {
    "HeadYaw": 0.0,
    "HeadPitch": 0.0,
    "LHipYawPitch": 0.0,
    "LHipRoll": 0.0,
    "LHipPitch": -0.45,
    "LKneePitch": 0.9,
    "LAnklePitch": -0.45,
    "LAnkleRoll": 0.0,
    "RHipYawPitch": 0.0,
    "RHipRoll": 0.0,
    "RHipPitch": -0.45,
    "RKneePitch": 0.9,
    "RAnklePitch": -0.45,
    "RAnkleRoll": 0.0,
    "LShoulderPitch": 1.4,
    "LShoulderRoll": 0.15,
    "LElbowYaw": -1.2,
    "LElbowRoll": -0.5,
    "LWristYaw": 0.0,
    "RShoulderPitch": 1.4,
    "RShoulderRoll": -0.15,
    "RElbowYaw": 1.2,
    "RElbowRoll": 0.5,
    "RWristYaw": 0.0,
    "LHand": 0.0,
    "RHand": 0.0,
    "LFinger.*": 0.0,
    "RFinger.*": 0.0,
    "LThumb.*": 0.0,
    "RThumb.*": 0.0,
}

NAO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=NAO_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # NAO torso sits ~0.33 m off the ground when standing; with bent knees
        # the GMR retarget keeps the torso around 0.39 m, so we initialize
        # slightly above that so the feet do not start in contact.
        pos=(0.0, 0.0, 0.40),
        joint_pos=NAO_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # Hip yaw-pitch + roll (medium torque)
        "hip_yaw_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*HipYawPitch", ".*HipRoll"],
            effort_limit_sim=3.348,
            velocity_limit_sim=4.16,
            stiffness=20.0,
            damping=1.0,
        ),
        # Hip pitch + knee + ankle pitch (carry the body weight)
        "leg_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*HipPitch", ".*KneePitch", ".*AnklePitch"],
            effort_limit_sim=3.023,
            velocity_limit_sim=6.40,
            stiffness=30.0,
            damping=1.5,
        ),
        # Ankle roll
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*AnkleRoll"],
            effort_limit_sim=3.348,
            velocity_limit_sim=4.16,
            stiffness=20.0,
            damping=1.0,
        ),
        # Arms (low torque, NAO arms are weak)
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*ShoulderPitch", ".*ShoulderRoll"],
            effort_limit_sim=1.783,
            velocity_limit_sim=8.27,
            stiffness=8.0,
            damping=0.4,
        ),
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*ElbowYaw", ".*ElbowRoll"],
            effort_limit_sim=1.547,
            velocity_limit_sim=7.34,
            stiffness=8.0,
            damping=0.4,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*WristYaw"],
            effort_limit_sim=0.4075,
            velocity_limit_sim=24.6,
            stiffness=2.0,
            damping=0.1,
        ),
        # Head
        "head": ImplicitActuatorCfg(
            joint_names_expr=["HeadYaw", "HeadPitch"],
            effort_limit_sim=1.547,
            velocity_limit_sim=8.27,
            stiffness=4.0,
            damping=0.2,
        ),
        # Hand grippers + fingers — passive, locked at zero with stiff PD so
        # they do not wander. They are deliberately excluded from the action
        # space in the env config.
        "passive_hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "LHand", "RHand",
                ".*Finger11", ".*Finger12", ".*Finger13",
                ".*Finger21", ".*Finger22", ".*Finger23",
                ".*Thumb1", ".*Thumb2",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=5.0,
            damping=0.2,
        ),
    },
)


# Mirrors the SPOT_ACTION_SCALE recipe: 0.25 * effort / stiffness, computed
# only over the 24 actively tracked joints. The env's joint_pos action is
# restricted to NAO_JOINT_NAMES, so passive hands do not need a scale.
def _build_action_scale() -> dict[str, float]:
    import re

    actuated_lookup: dict[str, tuple[float, float]] = {}
    for actuator in NAO_CFG.actuators.values():
        effort = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        for pattern in actuator.joint_names_expr:
            regex = re.compile(f"^{pattern}$")
            for joint in NAO_JOINT_NAMES:
                if regex.match(joint):
                    eff = effort[joint] if isinstance(effort, dict) else effort
                    stf = stiffness[joint] if isinstance(stiffness, dict) else stiffness
                    if stf:
                        actuated_lookup[joint] = (eff, stf)

    scale: dict[str, float] = {}
    for joint in NAO_JOINT_NAMES:
        if joint not in actuated_lookup:
            raise RuntimeError(f"NAO joint '{joint}' is not covered by any actuator group in NAO_CFG.")
        eff, stf = actuated_lookup[joint]
        scale[joint] = 0.25 * eff / stf
    return scale


NAO_ACTION_SCALE = _build_action_scale()
