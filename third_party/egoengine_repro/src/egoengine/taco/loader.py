"""TACO dataset loader for the EgoEngine digital-twin stage.

Unified per-frame access to one TACO egocentric sequence:
egocentric camera intrinsics/extrinsics, bimanual MANO hand poses,
target/tool object 6DoF poses + mesh paths, video metadata, and
cross-modality frame-alignment checks.

Data layout (dataset root, e.g. ``data/taco``)::

    Egocentric_RGB_Videos/<triplet>/<seq>/color.mp4
    Egocentric_Camera_Parameters/<triplet>/<seq>/egocentric_intrinsic.txt
    Egocentric_Camera_Parameters/<triplet>/<seq>/egocentric_frame_extrinsic.npy
    Hand_Poses/<triplet>/<seq>/{left,right}_hand{,_shape}.pkl
    Object_Poses/<triplet>/<seq>/{target,tool}_<objid>.npy
    object_models_released/<objid>_cm.obj          (shared, vertices in cm)

Frame conventions (verified 2026-07-27, see docs/TACO_LOADER.md):
- All modalities are indexed by the same frame axis, 0-based in arrays;
  hand-pkl keys are 5-digit strings "00001".. -> array index int(key)-1.
- Object poses are (N, 4, 4) object-to-world, meters.
- Hand translations are (N, 3) in the same world frame, meters.
- Camera extrinsics are (N, 4, 4) world-to-camera, OpenCV convention
  (x right, y down, z forward) — verified by projection overlay.
- Object mesh OBJs are in centimeters (suffix "_cm"); scale by 0.01.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import json
from dataclasses import dataclass, field
from os.path import join, isfile, isdir

import numpy as np

DEFAULT_DATASET_ROOT = join(os.path.dirname(__file__), "..", "..", "..", "data", "taco")
DEFAULT_DATASET_ROOT = os.path.abspath(DEFAULT_DATASET_ROOT)
DEFAULT_DATASET_ROOT = os.environ.get("TACO_ROOT", DEFAULT_DATASET_ROOT)
DEFAULT_MODEL_ROOT = os.path.abspath(
    join(os.path.dirname(__file__), "..", "..", "..", "data", "taco", "object_models_released")
)
DEFAULT_MODEL_ROOT = os.environ.get("TACO_MODEL_ROOT", join(DEFAULT_DATASET_ROOT, "object_models_released"))


@dataclass
class HandData:
    """Per-frame MANO parameters for one hand."""
    side: str
    pose: np.ndarray        # (N, 48) float32: axis-angle global_orient(3) + hand_pose(45)
    trans: np.ndarray       # (N, 3) float32, meters, world frame
    betas: np.ndarray       # (10,) float32
    frame_ids: np.ndarray   # (N,) int, 0-based array index of each entry


@dataclass
class ObjectData:
    """One manipulated object (tool or target)."""
    role: str               # "tool" | "target"
    object_id: str          # e.g. "071"
    name: str               # category from the triplet string (e.g. "brush")
    poses: np.ndarray       # (N, 4, 4) object-to-world, meters
    mesh_path: str          # OBJ, vertices in CENTIMETERS -> scale by 0.01


@dataclass
class TacoSequence:
    triplet: str
    seq_id: str
    root: str
    n_frames: int
    fps: float
    width: int
    height: int
    video_path: str
    intrinsic: np.ndarray               # (3, 3)
    extrinsics: np.ndarray              # (N, 4, 4) world-to-camera, OpenCV
    hands: dict = field(default_factory=dict)     # "left"/"right" -> HandData
    objects: dict = field(default_factory=dict)   # "tool"/"target" -> ObjectData

    # ------------------------------------------------------------------ API
    def read_frame(self, idx: int) -> np.ndarray:
        """Decode video frame `idx` (0-based) as RGB uint8 (H, W, 3)."""
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"could not decode frame {idx} of {self.video_path}")
        return bgr[:, :, ::-1].copy()

    def camera_position(self, idx: int) -> np.ndarray:
        """Camera center in world coordinates at frame `idx`."""
        R = self.extrinsics[idx, :3, :3]
        t = self.extrinsics[idx, :3, 3]
        return -R.T @ t

    def project(self, points_world: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Project (M, 3) world points to pixels at frame `idx`.

        Returns (uv (M, 2), z (M,)) — points with z <= 0 are behind the camera.
        OpenCV convention: x right, y down, z forward.
        """
        T = self.extrinsics[idx]
        pc = points_world @ T[:3, :3].T + T[:3, 3]
        z = pc[:, 2].copy()
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        uv = pc[:, :2] / safe[:, None]
        uv = uv * np.array([[self.intrinsic[0, 0], self.intrinsic[1, 1]]]) + np.array(
            [[self.intrinsic[0, 2], self.intrinsic[1, 2]]]
        )
        return uv, z

    def alignment_report(self) -> dict:
        """Per-modality frame counts and index bases."""
        report = {
            "triplet": self.triplet,
            "seq_id": self.seq_id,
            "video_frames": self.n_frames,
            "video_fps": self.fps,
            "camera_extrinsic_frames": int(self.extrinsics.shape[0]),
            "hands": {
                side: {
                    "frames": int(h.pose.shape[0]),
                    "frame_id_min": int(h.frame_ids.min()),
                    "frame_id_max": int(h.frame_ids.max()),
                    "contiguous": bool(np.array_equal(h.frame_ids, np.arange(h.frame_ids.size))),
                }
                for side, h in self.hands.items()
            },
            "objects": {
                role: {"frames": int(o.poses.shape[0]), "object_id": o.object_id, "name": o.name}
                for role, o in self.objects.items()
            },
        }
        counts = {self.n_frames, self.extrinsics.shape[0]}
        counts.update(h.pose.shape[0] for h in self.hands.values())
        counts.update(o.poses.shape[0] for o in self.objects.values())
        report["aligned"] = len(counts) == 1
        return report


# ---------------------------------------------------------------------- load
def parse_triplet(triplet: str) -> tuple[str, str, str]:
    """"(action, tool, target)" -> (action, tool_name, target_name).

    TACO's object_additional_information is only a purchase list (category x
    shopping links); the human-readable category of the tool/target objects
    comes from the triplet string itself.
    """
    parts = [p.strip() for p in triplet.strip().strip("()").split(",")]
    if len(parts) != 3:
        raise ValueError(f"bad triplet {triplet!r}")
    return parts[0], parts[1], parts[2]


def _probe_video(video_path: str) -> tuple[int, float, int, int]:
    """(frame_count, fps, width, height). ffprobe first (exact), cv2 fallback."""
    # On a shared cluster filesystem, ffprobe --count_frames scans the complete
    # 1080p file and can spend many minutes before Isaac even starts.  The
    # aligned TACO arrays provide the authoritative frame count; this fast
    # metadata path is enabled by the runtime wrapper for action-branch jobs.
    if os.environ.get("EGOENGINE_FAST_VIDEO_PROBE") == "1":
        try:
            raw = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames,r_frame_rate,width,height",
                    "-of", "json", video_path,
                ],
                capture_output=True, text=True, check=True, timeout=20,
            ).stdout
            stream = json.loads(raw)["streams"][0]
            rate = stream["r_frame_rate"].split("/")
            fps = float(rate[0]) / float(rate[1])
            raw_n = stream.get("nb_frames")
            n = int(raw_n) if raw_n not in (None, "N/A") else -1
            return n, fps, int(stream["width"]), int(stream["height"])
        except Exception:
            # Do not fall through to the full-file frame scan in the fast
            # action-branch path.  Frame count is inferred from the aligned
            # simulator arrays below; video dimensions are unused there.
            return -1, 30.0, 0, 0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")
        # csv order follows the requested entries
        w, h = int(out[0]), int(out[1])
        num, den = out[2].split("/")
        fps = float(num) / float(den)
        n = int(out[3])
        return n, fps, w, h
    except Exception:
        import cv2
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return n, fps, w, h


def _load_hand(hand_dir: str, side: str) -> HandData:
    pose_path = join(hand_dir, f"{side}_hand.pkl")
    shape_path = join(hand_dir, f"{side}_hand_shape.pkl")
    data = pickle.load(open(pose_path, "rb"))
    keys = sorted(data.keys())  # "00001" .. "NNNNN"
    frame_ids = np.array([int(k) - 1 for k in keys], dtype=np.int64)
    pose = np.stack([np.asarray(data[k]["hand_pose"].detach().cpu(), dtype=np.float32) for k in keys])
    trans = np.stack([np.asarray(data[k]["hand_trans"].detach().cpu(), dtype=np.float32) for k in keys])
    if isfile(shape_path):
        betas = np.asarray(
            pickle.load(open(shape_path, "rb"))["hand_shape"].detach().cpu(), dtype=np.float32
        ).reshape(10)
    else:
        betas = np.zeros(10, dtype=np.float32)
    return HandData(side=side, pose=pose, trans=trans, betas=betas, frame_ids=frame_ids)


def _load_objects(pose_dir: str, model_root: str, names: dict) -> dict:
    objects = {}
    for fn in sorted(os.listdir(pose_dir)):
        if not fn.endswith(".npy"):
            continue
        stem = fn[:-4]
        if stem.startswith("tool_"):
            role = "tool"
        elif stem.startswith("target_"):
            role = "target"
        else:
            continue
        object_id = stem.split("_", 1)[1]
        poses = np.load(join(pose_dir, fn)).astype(np.float32)
        assert poses.ndim == 3 and poses.shape[1:] == (4, 4), f"unexpected pose shape {poses.shape} in {fn}"
        mesh_path = join(model_root, f"{object_id}_cm.obj")
        if not isfile(mesh_path):
            raise FileNotFoundError(f"object mesh missing: {mesh_path}")
        objects[role] = ObjectData(
            role=role, object_id=object_id,
            name=names.get(role, object_id),
            poses=poses, mesh_path=mesh_path,
        )
    return objects


def load_sequence(triplet: str, seq_id: str,
                  root: str = DEFAULT_DATASET_ROOT,
                  model_root: str = DEFAULT_MODEL_ROOT) -> TacoSequence:
    """Load one TACO egocentric sequence into a unified per-frame structure."""
    cam_dir = join(root, "Egocentric_Camera_Parameters", triplet, seq_id)
    hand_dir = join(root, "Hand_Poses", triplet, seq_id)
    pose_dir = join(root, "Object_Poses", triplet, seq_id)
    video_path = join(root, "Egocentric_RGB_Videos", triplet, seq_id, "color.mp4")
    for p in (cam_dir, hand_dir, pose_dir, video_path):
        if not (isdir(p) or isfile(p)):
            raise FileNotFoundError(p)

    intrinsic = np.loadtxt(join(cam_dir, "egocentric_intrinsic.txt")).astype(np.float32)
    extrinsics = np.load(join(cam_dir, "egocentric_frame_extrinsic.npy")).astype(np.float32)
    hands = {side: _load_hand(hand_dir, side) for side in ("left", "right")}
    _, tool_name, target_name = parse_triplet(triplet)
    objects = _load_objects(pose_dir, model_root, {"tool": tool_name, "target": target_name})
    n_frames, fps, width, height = _probe_video(video_path)
    if n_frames < 0:
        # TACO modalities are frame-aligned and are already loaded above.
        n_frames = max(
            [extrinsics.shape[0]]
            + [hand.pose.shape[0] for hand in hands.values()]
            + [obj.poses.shape[0] for obj in objects.values()]
        )

    return TacoSequence(
        triplet=triplet, seq_id=seq_id, root=root,
        n_frames=n_frames, fps=fps, width=width, height=height,
        video_path=video_path, intrinsic=intrinsic, extrinsics=extrinsics,
        hands=hands, objects=objects,
    )
