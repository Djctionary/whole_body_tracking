import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


SPOT_URDF_PATH = "/home/vergil/MENU/Projects/AnimaSpot/urdf/isaacsim_spot/spot.urdf"

SPOT_JOINT_NAMES = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]

SPOT_FOOT_NAMES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]

SPOT_TRACKING_BODY_NAMES = [
    "spot",
    "fl_hip",
    "fl_uleg",
    "fl_lleg",
    "fr_hip",
    "fr_uleg",
    "fr_lleg",
    "hl_hip",
    "hl_uleg",
    "hl_lleg",
    "hr_hip",
    "hr_uleg",
    "hr_lleg",
]

SPOT_STANDING_JOINT_POS = {
    "fl_hx": 0.0,
    "fl_hy": 0.85,
    "fl_kn": -1.65,
    "fr_hx": 0.0,
    "fr_hy": 0.85,
    "fr_kn": -1.65,
    "hl_hx": 0.0,
    "hl_hy": 0.95,
    "hl_kn": -1.7,
    "hr_hx": 0.0,
    "hr_hy": 0.95,
    "hr_kn": -1.7,
}

SPOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=SPOT_URDF_PATH,
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos=SPOT_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hx", ".*_hy"],
            effort_limit_sim={".*_hx": 45.0, ".*_hy": 45.0},
            velocity_limit_sim={".*_hx": 100.0, ".*_hy": 100.0},
            stiffness={".*_hx": 40.0, ".*_hy": 55.0},
            damping={".*_hx": 4.0, ".*_hy": 5.5},
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_kn"],
            effort_limit_sim={".*_kn": 115.0},
            velocity_limit_sim={".*_kn": 100.0},
            stiffness={".*_kn": 80.0},
            damping={".*_kn": 8.0},
        ),
    },
)

SPOT_ACTION_SCALE = {}
for actuator in SPOT_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in names}
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            SPOT_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
