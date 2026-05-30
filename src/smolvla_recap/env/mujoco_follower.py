"""MuJoCo-based follower that replaces the physical SO-101 follower arm.

Accepts normalized joint commands from the leader arm ([-100, 100] for body
joints, [0, 100] for gripper) and drives a MuJoCo simulation, rendering
camera images for dataset recording.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_XML = ASSET_DIR / "so101_tabletop.xml"

# Joint names in actuator order (matches SO-101 motor layout).
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits in radians, taken from the MuJoCo XML.
# Each tuple is (min_rad, max_rad).
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.9199, 1.9199),
    "shoulder_lift": (-1.7453, 1.7453),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.6581, 1.6581),
    "wrist_roll": (-2.7438, 2.8412),
    "gripper": (-0.1745, 1.7453),
}

# Constant radian offset applied *after* norm→rad conversion for send_action
# and subtracted *before* rad→norm conversion for get_observation.
# Corrects the 90° orientation mismatch between the real and sim wrist_roll.
JOINT_OFFSETS: dict[str, float] = {
    "wrist_roll": -np.pi / 2,
}

# Number of substeps per control step.  The model timestep is 0.002 s so
# 17 substeps ≈ one 30 Hz control step (0.034 s ≈ 1/30).
_DEFAULT_SUBSTEPS = 17


class MuJoCoFollower:
    """Drop-in replacement for a physical SO-101 follower arm.

    Provides the same high-level interface used by the recording loop:
    ``connect``, ``get_observation``, ``send_action``, ``disconnect``.
    """

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        camera_names: list[str] | None = None,
        image_size: tuple[int, int] = (480, 640),
        substeps: int = _DEFAULT_SUBSTEPS,
    ):
        self.xml_path = Path(xml_path)
        self.camera_names = camera_names or ["front_camera", "wrist_camera"]
        self.image_height, self.image_width = image_size
        self.substeps = substeps

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._viewer = None
        self._is_connected = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, keyframe: str = "ready", viewer: bool = False) -> None:
        """Load the MuJoCo model, create data, and initialise renderers.

        Args:
            keyframe: Name of the keyframe to initialise the arm pose from.
            viewer: If ``True``, open an interactive viewer window so you can
                    see the simulation while teleoperating.
        """
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Apply a keyframe so the arm starts in a known pose.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        mujoco.mj_forward(self.model, self.data)

        # Create one offscreen renderer per camera.
        for cam_name in self.camera_names:
            renderer = mujoco.Renderer(self.model, self.image_height, self.image_width)
            self._renderers[cam_name] = renderer

        # Optionally launch a passive (non-blocking) viewer window.
        if viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self._is_connected = True

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return current joint positions (normalized) and rendered camera images.

        Returns a dict with keys:
            ``"state"``  – float32 array of shape (6,), normalized joint positions
            ``"<camera_name>"`` – uint8 array of shape (H, W, 3) for each camera
        """
        self._check_connected()

        # --- Joint positions (radians -> normalized) ---
        state = np.array(
            [
                self._rad_to_norm(
                    name,
                    self.data.qpos[self._joint_qpos_idx(name)] - JOINT_OFFSETS.get(name, 0.0),
                )
                for name in JOINT_NAMES
            ],
            dtype=np.float32,
        )

        obs: dict[str, np.ndarray] = {"state": state}

        # --- Camera images ---
        for cam_name, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=cam_name)
            obs[cam_name] = renderer.render().copy()

        return obs

    def send_action(self, action: dict[str, float]) -> None:
        """Set actuator controls from normalized leader values, then step the sim.

        Args:
            action: mapping from ``"<joint_name>.pos"`` to a normalized value
                    ([-100, 100] for body joints, [0, 100] for gripper).
        """
        self._check_connected()

        for name in JOINT_NAMES:
            key = f"{name}.pos"
            if key not in action:
                continue
            rad = self._norm_to_rad(name, action[key])
            rad += JOINT_OFFSETS.get(name, 0.0)
            lo, hi = JOINT_LIMITS[name]
            rad = float(np.clip(rad, lo, hi))
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.data.ctrl[act_id] = rad

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    # Fixed cube positions for evaluation.
    # Default is the training position; the others are small offsets.
    CUBE_POSITIONS = [
        (-0.10, -0.05),   # default (training position)
        (-0.13, -0.05),   # 3 cm to the left
        (-0.10, -0.02),   # 3 cm forward (toward centre of table)
    ]

    def reset(
        self,
        keyframe: str = "ready",
        randomize_cube: bool = False,
        cube_position_idx: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Reset the full scene (arm pose, object positions, velocities) to a keyframe.

        Args:
            keyframe: Name of the MuJoCo keyframe to reset to.
            randomize_cube: If ``True``, cycle through the fixed
                ``CUBE_POSITIONS`` list using *cube_position_idx*, or pick
                one at random if *cube_position_idx* is ``None``.
            cube_position_idx: Index into ``CUBE_POSITIONS`` to use.
                If ``None``, a random position is chosen.
            rng: Optional numpy random generator for reproducibility.
        """
        self._check_connected()

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id == -1:
            raise ValueError(f"Keyframe '{keyframe}' not found in model")

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        if randomize_cube:
            if cube_position_idx is not None:
                idx = cube_position_idx % len(self.CUBE_POSITIONS)
            else:
                if rng is None:
                    rng = np.random.default_rng()
                idx = int(rng.integers(len(self.CUBE_POSITIONS)))
            cx, cy = self.CUBE_POSITIONS[idx]
            # Cube freejoint is the first joint → qpos[0:7] = (x,y,z, qw,qx,qy,qz)
            self.data.qpos[0] = cx
            self.data.qpos[1] = cy
            # z and quaternion stay at keyframe defaults
            print(f"  Cube position {idx}: ({cx:.3f}, {cy:.3f})")

        mujoco.mj_forward(self.model, self.data)

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def disconnect(self) -> None:
        """Release renderer and viewer resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self.model = None
        self.data = None
        self._is_connected = False

    # ------------------------------------------------------------------
    # Value mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_to_rad(joint_name: str, norm_val: float) -> float:
        """Convert a normalized leader value to MuJoCo radians.

        Body joints: norm in [-100, 100]  ->  rad = min + (norm + 100)/200 * (max - min)
        Gripper:     norm in [0, 100]     ->  rad = min + norm/100 * (max - min)
        """
        lo, hi = JOINT_LIMITS[joint_name]
        if joint_name == "gripper":
            norm_val = float(np.clip(norm_val, 0, 100))
            return lo + (norm_val / 100.0) * (hi - lo)
        else:
            norm_val = float(np.clip(norm_val, -100, 100))
            return lo + ((norm_val + 100.0) / 200.0) * (hi - lo)

    @staticmethod
    def _rad_to_norm(joint_name: str, rad_val: float) -> float:
        """Convert MuJoCo radians back to normalized leader value."""
        lo, hi = JOINT_LIMITS[joint_name]
        if joint_name == "gripper":
            return float(np.clip((rad_val - lo) / (hi - lo) * 100.0, 0, 100))
        else:
            return float(np.clip((rad_val - lo) / (hi - lo) * 200.0 - 100.0, -100, 100))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _joint_qpos_idx(self, joint_name: str) -> int:
        """Return the qpos index for a named joint."""
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return self.model.jnt_qposadr[jnt_id]

    def _check_connected(self) -> None:
        if not self._is_connected:
            raise RuntimeError("MuJoCoFollower is not connected. Call connect() first.")
