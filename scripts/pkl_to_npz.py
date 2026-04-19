"""Convert a GMR-retargeted .pkl motion into the .npz schema consumed by
``whole_body_tracking``'s training-time MotionLoader.

The .pkl produced by ``GMR/scripts/smplx_to_robot.py`` contains:

- ``fps``: int (typically 30 for BEAT2)
- ``root_pos``: ``(T, 3)`` float, world translation in meters (Z-up)
- ``root_rot``: ``(T, 4)`` float, **xyzw** quaternion
- ``dof_pos``: ``(T, n_dof)`` float, joint angles in MJCF qpos order (after
  the free joint). For NAO this has 40 columns; the 16 finger columns are
  constant zero in current GMR retargets.

The output .npz must contain the same keys as ``csv_to_npz.py`` writes:
``fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
body_ang_vel_w`` — all derived from the IsaacLab simulator after writing
the retargeted root + joint state into the scene articulation.

.. code-block:: bash

    # Usage
    python pkl_to_npz.py \
        --input_file /path/to/10_kieks_0_10_10_nao.pkl \
        --output_name 10_kieks_0_10_10_nao \
        --output_file /path/to/10_kieks_0_10_10_nao.npz \
        --robot nao --output_fps 50 --disable_wandb
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
from dataclasses import MISSING

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a retargeted .pkl motion to the training-time .npz format.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input retargeted .pkl file.")
parser.add_argument(
    "--input_fps",
    type=int,
    default=0,
    help="Override the input fps. 0 means use the value stored in the pkl (recommended).",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range: START END (both inclusive, 1-indexed). If omitted, use all frames.",
)
parser.add_argument("--output_name", type=str, required=True, help="Logical motion name (used for W&B artifact).")
parser.add_argument("--output_fps", type=int, default=50, help="Output fps for the resampled motion.")
parser.add_argument(
    "--robot",
    type=str,
    choices=("nao",),
    default="nao",
    help="Robot configuration to use (currently only NAO is supported by this converter).",
)
parser.add_argument(
    "--output_file", type=str, default="/tmp/motion.npz", help="Local path for the exported motion NPZ."
)
parser.add_argument(
    "--disable_wandb", action="store_true", help="Skip uploading the generated motion to Weights & Biases."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.nao import NAO_CFG, NAO_PKL_DOF_ORDER

ROBOT_CONFIGS = {
    "nao": {"cfg": NAO_CFG, "pkl_dof_order": NAO_PKL_DOF_ORDER},
}


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Tolerate pkl files written with a newer numpy (>=2.0) that pickled the
    private ``numpy._core.*`` paths, on environments where only ``numpy.core.*``
    exists (or vice versa). Both module trees expose the same public classes
    such as ``ndarray`` and ``dtype``, so a simple module-name remap is safe.
    """

    _PREFIX_REMAP = {
        "numpy._core": "numpy.core",
        "numpy.core": "numpy._core",
    }

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            for src, dst in self._PREFIX_REMAP.items():
                if module == src or module.startswith(src + "."):
                    remapped = dst + module[len(src):]
                    return super().find_class(remapped, name)
            raise


def _safe_pickle_load(file_obj):
    return _NumpyCompatUnpickler(file_obj).load()


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = MISSING


class PklMotionLoader:
    """Loads a GMR-retargeted .pkl, then resamples / interpolates to ``output_fps``.

    Mirrors the behavior of ``csv_to_npz.MotionLoader`` but takes pkl input
    and reorders the root quaternion from xyzw to wxyz.
    """

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        expected_dofs: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.expected_dofs = expected_dofs
        self.device = device
        self.frame_range = frame_range
        self._load_motion(input_fps)
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.duration = (self.input_frames - 1) * self.input_dt
        self.current_idx = 0
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self, input_fps_override: int):
        with open(self.motion_file, "rb") as f:
            data = _safe_pickle_load(f)

        for key in ("fps", "root_pos", "root_rot", "dof_pos"):
            if key not in data:
                raise KeyError(f"Required key '{key}' missing from pkl '{self.motion_file}'.")

        pkl_fps = int(data["fps"])
        self.input_fps = int(input_fps_override) if input_fps_override > 0 else pkl_fps

        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must be (T, 3); got {root_pos.shape}.")
        if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
            raise ValueError(f"root_rot must be (T, 4); got {root_rot_xyzw.shape}.")
        if dof_pos.ndim != 2 or dof_pos.shape[1] != self.expected_dofs:
            raise ValueError(
                f"dof_pos must be (T, {self.expected_dofs}); got {dof_pos.shape}. "
                "Check that the pkl matches the configured robot."
            )
        if not (root_pos.shape[0] == root_rot_xyzw.shape[0] == dof_pos.shape[0]):
            raise ValueError("root_pos, root_rot and dof_pos must have the same number of frames.")

        # GMR stores the quaternion as xyzw; IsaacLab uses wxyz.
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

        if self.frame_range is not None:
            start, end = self.frame_range
            start_idx = max(0, start - 1)
            end_idx = min(root_pos.shape[0], end)
            root_pos = root_pos[start_idx:end_idx]
            root_rot_wxyz = root_rot_wxyz[start_idx:end_idx]
            dof_pos = dof_pos[start_idx:end_idx]

        self.motion_base_poss_input = torch.from_numpy(root_pos).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(root_rot_wxyz).to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(self.device)
        self.input_frames = root_pos.shape[0]

        print(
            f"[INFO] PKL loaded: {self.motion_file}\n"
            f"       input fps: {self.input_fps} (pkl reports {pkl_fps}),"
            f" input frames: {self.input_frames},"
            f" duration: {(self.input_frames - 1) / self.input_fps:.3f} s,"
            f" dofs: {self.motion_dof_poss_input.shape[1]}"
        )

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1)
        )
        print(
            f"[INFO] Motion interpolated: input frames={self.input_frames} @ {self.input_fps} Hz -> "
            f"output frames={self.output_frames} @ {self.output_fps} Hz"
        )

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, pkl_dof_order: list[str]):
    motion = PklMotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        expected_dofs=len(pkl_dof_order),
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range else None,
    )

    robot = scene["robot"]
    # Map pkl dof_pos columns -> articulation joint indices by name.
    robot_joint_indexes = robot.find_joints(pkl_dof_order, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([1.5, 1.5, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez(args_cli.output_file, **log)
            print(f"[INFO] Motion saved locally to: {args_cli.output_file}")
            print(
                f"[INFO] joint_pos.shape={log['joint_pos'].shape},"
                f" body_pos_w.shape={log['body_pos_w'].shape},"
                f" frames={log['joint_pos'].shape[0]}"
            )

            if not args_cli.disable_wandb:
                import wandb

                collection = args_cli.output_name
                run = wandb.init(project="pkl_to_npz", name=collection)
                print(f"[INFO] Logging motion to wandb: {collection}")
                registry = "motions"
                logged_artifact = run.log_artifact(
                    artifact_or_path=args_cli.output_file, name=collection, type=registry
                )
                run.link_artifact(
                    artifact=logged_artifact, target_path=f"wandb-registry-{registry}/{collection}"
                )
                print(f"[INFO] Motion saved to wandb registry: {registry}/{collection}")

            print("[INFO] Conversion finished, exiting simulator.")
            simulation_app.close()
            return


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    robot_config = ROBOT_CONFIGS[args_cli.robot]
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_config["cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, pkl_dof_order=robot_config["pkl_dof_order"])


if __name__ == "__main__":
    main()
    simulation_app.close()
