"""Project LIBERO Panda gripper finger tips to agentview pixels from RLDS proprio / sim FK."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

_OPENVLA_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _OPENVLA_ROOT.parent

# LIBERO agentview 256x256 camera transform (robosuite, fixed across spatial tasks)
_AGENTVIEW_T = np.array(
    [
        [-99.58379766, 309.01931651, -80.41815266, 195.08857519],
        [94.56270613, -4.19308495e-05, -320.83460633, 454.37577193],
        [-0.77799824, -1.52115266e-07, -0.62826645, 1.52412879],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_FINGER_X = 0.014
_FINGER_BODY_NAMES = ("gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip")


def _project_points_from_world_to_camera(
    points: np.ndarray,
    world_to_camera_transform: np.ndarray,
    camera_height: int,
    camera_width: int,
) -> np.ndarray:
    assert points.shape[-1] == 3
    ones_pad = np.ones(points.shape[:-1] + (1,))
    hom = np.concatenate((points, ones_pad), axis=-1)
    mat = world_to_camera_transform.reshape([1] * (hom.ndim - 1) + [4, 4])
    pixels = np.matmul(mat, hom[..., None])[..., 0]
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2].round().astype(int)
    return np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )


def _project_world_to_rlds_pixel(
    world_xyz: np.ndarray,
    T: np.ndarray,
    height: int = 256,
    width: int = 256,
) -> Tuple[int, int]:
    row, col = _project_points_from_world_to_camera(world_xyz[None], T, height, width)[0]
    x = width - 1 - int(col)
    y = int(row)
    return x, y


def _finger_local_offsets(gripper_qpos: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    g0, g1 = float(gripper_qpos[0]), float(gripper_qpos[1])
    z0 = -0.015 - 0.75 * abs(g0)
    z1 = -0.015 - 0.75 * abs(g1)
    return np.array([_FINGER_X, g0, z0]), np.array([_FINGER_X, g1, z1])


def _sim_python() -> Optional[str]:
    for cand in (
        os.environ.get("GRIPPER_SIM_PYTHON"),
        os.environ.get("SUBOPT_PYTHON"),
        shutil.which("python"),
        "/home/fan-test/miniconda3/envs/subopt/bin/python",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


class _InProcessSimGripper:
    """Lazy LIBERO env for direct sim FK (robosuite available in current interpreter)."""

    _env = None
    _lock = threading.Lock()

    @classmethod
    def _ensure(cls):
        if cls._env is not None:
            return cls._env
        libero_root = _REPO_ROOT / "LIBERO"
        if libero_root.exists() and str(libero_root) not in sys.path:
            sys.path.insert(0, str(libero_root))
        from libero.libero.envs import OffScreenRenderEnv
        from libero_sim_mask import gripper_finger_pixels

        spatial = _REPO_ROOT / "LIBERO/libero/libero/bddl_files/libero_spatial"
        bddl = os.environ.get("GRIPPER_SIM_BDDL")
        if not bddl or not Path(bddl).exists():
            bddl = str(sorted(spatial.glob("*.bddl"))[0])
        cls._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
        )
        cls._env.reset()
        cls._finger_fn = gripper_finger_pixels
        return cls._env

    @classmethod
    def pixels(
        cls,
        joint_state: Sequence[float],
        gripper_qpos: Sequence[float],
        *,
        height: int = 256,
        width: int = 256,
    ) -> List[Tuple[int, int]]:
        with cls._lock:
            env = cls._ensure()
            sim = env.env.sim
            js = np.asarray(joint_state, dtype=np.float64).reshape(-1)
            g = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
            sim.data.qpos[:7] = js[:7]
            if sim.model.nq >= 9:
                sim.data.qpos[7:9] = g[:2]
            sim.forward()
            return cls._finger_fn(sim, height=height, width=width)


class _SubprocessSimGripper:
    """Persistent subopt worker when robosuite is not in the current env."""

    _proc: Optional[subprocess.Popen] = None
    _lock = threading.Lock()

    @classmethod
    def _start(cls):
        py = _sim_python()
        if py is None:
            raise RuntimeError("GRIPPER_SIM_PYTHON / subopt python not found")
        worker = _OPENVLA_ROOT / "gripper_sim_worker.py"
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        cls._proc = subprocess.Popen(
            [py, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        ready_line = cls._proc.stdout.readline()
        if not ready_line:
            err = cls._proc.stderr.read() if cls._proc.stderr else ""
            raise RuntimeError(f"gripper sim worker failed to start: {err}")
        info = json.loads(ready_line)
        if not info.get("ready"):
            raise RuntimeError(f"gripper sim worker bad handshake: {info}")

    @classmethod
    def pixels(
        cls,
        joint_state: Sequence[float],
        gripper_qpos: Sequence[float],
        *,
        height: int = 256,
        width: int = 256,
    ) -> List[Tuple[int, int]]:
        with cls._lock:
            if cls._proc is None or cls._proc.poll() is not None:
                cls._start()
            req = {
                "joint": [float(x) for x in np.asarray(joint_state).reshape(-1)[:7]],
                "gripper": [float(x) for x in np.asarray(gripper_qpos).reshape(-1)[:2]],
                "h": height,
                "w": width,
            }
            assert cls._proc.stdin is not None and cls._proc.stdout is not None
            cls._proc.stdin.write(json.dumps(req) + "\n")
            cls._proc.stdin.flush()
            line = cls._proc.stdout.readline()
            if not line:
                err = cls._proc.stderr.read() if cls._proc.stderr else ""
                raise RuntimeError(f"gripper sim worker died: {err}")
            resp = json.loads(line)
            return [(int(x), int(y)) for x, y in resp.get("pts", [])]


def _can_use_inprocess_sim() -> bool:
    try:
        import robosuite  # noqa: F401
    except ImportError:
        return False
    libero_root = _REPO_ROOT / "LIBERO"
    return libero_root.exists()


def gripper_pixels_from_joint(
    joint_state: Sequence[float],
    gripper_qpos: Sequence[float],
    *,
    height: int = 256,
    width: int = 256,
) -> List[Tuple[int, int]]:
    """Sim FK: set Panda qpos from RLDS joint_state + gripper, read finger-tip pixels."""
    try:
        if _can_use_inprocess_sim():
            return _InProcessSimGripper.pixels(
                joint_state, gripper_qpos, height=height, width=width
            )
        return _SubprocessSimGripper.pixels(
            joint_state, gripper_qpos, height=height, width=width
        )
    except Exception:
        return []


def gripper_pixels_from_state(
    state: Sequence[float],
    *,
    height: int = 256,
    width: int = 256,
    camera_T: np.ndarray | None = None,
) -> List[Tuple[int, int]]:
    """Analytic EEF-offset fallback (less accurate than sim FK)."""
    st = np.asarray(state, dtype=np.float64).reshape(-1)
    if st.shape[0] < 8:
        return []

    eef_pos = st[:3]
    rot = R.from_rotvec(st[3:6])
    T = _AGENTVIEW_T if camera_T is None else camera_T
    locals_ = _finger_local_offsets(st[6:8])

    pts: List[Tuple[int, int]] = []
    for local in locals_:
        world = eef_pos + rot.apply(local)
        x, y = _project_world_to_rlds_pixel(world, T, height, width)
        if 0 <= x < width and 0 <= y < height:
            pts.append((x, y))
    return pts


def gripper_pixels_from_obs(
    state: Optional[Sequence[float]] = None,
    *,
    joint_state: Optional[Sequence[float]] = None,
    height: int = 256,
    width: int = 256,
    prefer_sim: bool = True,
) -> List[Tuple[int, int]]:
    """
    Preferred: sim FK from joint_state (+ gripper qpos from state[6:8]).
    Fallback: analytic projection from 8D proprio.
    """
    if prefer_sim and joint_state is not None:
        gq = None
        if state is not None:
            st = np.asarray(state, dtype=np.float64).reshape(-1)
            if st.shape[0] >= 8:
                gq = st[6:8]
        if gq is None:
            js = np.asarray(joint_state, dtype=np.float64).reshape(-1)
            if js.shape[0] >= 7:
                gq = js[6:8] if js.shape[0] >= 9 else (0.04, -0.04)
        if gq is not None:
            pts = gripper_pixels_from_joint(
                joint_state, gq, height=height, width=width
            )
            if pts:
                return pts
    if state is not None:
        return gripper_pixels_from_state(state, height=height, width=width)
    return []


def draw_gripper_dots_on_rgb(
    rgb: np.ndarray,
    state: Sequence[float],
    *,
    joint_state: Optional[Sequence[float]] = None,
    radius: int | None = None,
) -> np.ndarray:
    """Draw white finger-tip dots on masked RGB image (in-place safe copy)."""
    from libero_sim_mask import draw_white_dots

    pts = gripper_pixels_from_obs(state, joint_state=joint_state, height=rgb.shape[0], width=rgb.shape[1])
    if not pts:
        return rgb
    H, W = rgb.shape[:2]
    if radius is None:
        radius = max(2, min(5, min(H, W) // 70))
    out = rgb.copy() if not rgb.flags.writeable else rgb
    return draw_white_dots(out, pts, radius=radius)
