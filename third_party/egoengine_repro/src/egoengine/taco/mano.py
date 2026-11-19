"""MANO mesh computation for TACO hand poses, via smplx.

Matches the semantics of TACO-Instructions' dataset_utils/hand_pose_loader.py
(manopth, use_pca=False, ncomps=45, flat_hand_mean default True, center_idx=0):
- input hand_pose (48,) = axis-angle global orientation (3) + full 45-dim
  articulation (no PCA);
- input hand_trans (3,) in meters, world frame — it is the WRIST position
  (manopth centers the model on joint 0 before trans is added);
- output vertices (778, 3) and joints (21, 3) in METERS, world frame
  (manopth returns mm and its loader divides by 1000 before adding trans;
  smplx returns meters directly).

Joint order follows the manopth reordering (TACO convention):
0 wrist; 1-4 thumb (3 joints + tip); 5-8 index; 9-12 middle; 13-16 ring;
17-20 pinky. smplx only returns the 16 kinematic joints — the 5 fingertip
joints are taken from mesh vertices (smplx VERTEX_IDS['mano']).
"""

from __future__ import annotations

import os
from os.path import join

import numpy as np

# fingertip vertex ids (thumb, index, middle, ring, pinky), smplx convention
TIP_VERTEX_IDS = [744, 320, 443, 554, 671]
# manopth reordering: [wrist, thumb chain+tip, index, middle, ring, pinky]
MANOPTH_JOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# smplx expects <root>/mano/MANO_{LEFT,RIGHT}.pkl; data/models/mano_smplx/mano
# symlinks to the shared mano_v1_2/models directory.
DEFAULT_MANO_ROOT = os.path.abspath(
    join(os.path.dirname(__file__), "..", "..", "..", "data", "models", "mano_smplx")
)
DEFAULT_MANO_ROOT = os.environ.get("MANO_ROOT", DEFAULT_MANO_ROOT)

_layer_cache: dict = {}


def _install_chumpy_stub() -> None:
    """MANO .pkl files contain chumpy.Ch arrays; stub chumpy with an ndarray
    subclass so pickle can load them (chumpy itself is numpy-2-incompatible)."""
    import sys
    import types

    if "chumpy" in sys.modules:
        return
    try:
        import chumpy  # noqa: F401
        return
    except ImportError:
        pass

    class Ch(np.ndarray):
        """ndarray subclass tolerant of chumpy's pickle patterns."""

        def __new__(cls, *args, **kwargs):
            if args and isinstance(args[0], np.ndarray):
                return args[0].view(cls)
            if args and isinstance(args[0], (tuple, list)) and len(args[0]) > 0:
                return np.empty(tuple(args[0])).view(cls)
            return np.zeros(0).view(cls)

        def __setstate__(self, state):
            if isinstance(state, dict):
                # chumpy stores extra attributes (e.g. sparse blocks) as a dict
                self.__dict__.update(state)
            else:
                super().__setstate__(state)

    class Select:
        """Eager stand-in for chumpy.reordering.Select: the MANO pkl stores
        shapedirs/posedirs as lazy row/col selections; smplx needs real
        arrays, so evaluate the selection (a[r][:, c]) on demand."""

        def __init__(self, *args, **kwargs):
            if args:
                self.a = args[0]
            if len(args) > 1:
                self.r = args[1]
            if len(args) > 2:
                self.c = args[2]
            self.kwargs = kwargs

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        def _eval(self):
            d = self.__dict__
            a = d.get("a", np.zeros(0))
            # chumpy wraps the payload array in a Ch whose data sits in `.x`
            a = np.asarray(getattr(a, "x", a))
            if "idxs" in d:
                out = a.ravel()[np.asarray(d["idxs"])]
                if "preferred_shape" in d:
                    out = out.reshape(d["preferred_shape"])
                return np.ascontiguousarray(out)
            r = d.get("r", slice(None))
            c = d.get("c", slice(None))
            out = a[r]
            if out.ndim > 1:
                out = out[:, c]
            elif not isinstance(c, slice):
                out = out[c]
            return np.ascontiguousarray(out)

        @property
        def shape(self):
            return self._eval().shape

        @property
        def ndim(self):
            return self._eval().ndim

        @property
        def dtype(self):
            return self._eval().dtype

        def __array__(self, dtype=None, copy=None):
            arr = self._eval()
            return arr.astype(dtype) if dtype is not None else arr

        def __getitem__(self, item):
            return self._eval()[item]

    chumpy = types.ModuleType("chumpy")
    chumpy_ch = types.ModuleType("chumpy.ch")
    chumpy_reordering = types.ModuleType("chumpy.reordering")
    chumpy.Ch = Ch
    chumpy_ch.Ch = Ch
    chumpy_reordering.Select = Select
    chumpy.ch = chumpy_ch
    chumpy.reordering = chumpy_reordering
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy_ch
    sys.modules["chumpy.reordering"] = chumpy_reordering


def get_mano_layer(side: str, mano_root: str = DEFAULT_MANO_ROOT, device: str = "cpu"):
    """Cached smplx MANO layer for one side ('left'/'right')."""
    key = (side, mano_root, device)
    if key not in _layer_cache:
        _install_chumpy_stub()
        import smplx
        import torch
        layer = smplx.create(
            model_path=mano_root,
            model_type="mano",
            is_rhand=(side == "right"),
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        )
        layer = layer.to(device)
        _layer_cache[key] = layer
    return _layer_cache[key]


def mano_forward(hand, mano_root: str = DEFAULT_MANO_ROOT, device: str = "cpu",
                 chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame MANO vertices + joints from a HandData record.

    Args:
        hand: egoengine.taco.loader.HandData (pose (N,48), trans (N,3), betas (10,)).
        chunk: batch size per forward pass (CPU memory guard).

    Returns:
        vertices (N, 778, 3) float32, joints (N, 21, 3) float32 (meters, world),
        faces (F, 3) int32.
    """
    import torch

    layer = get_mano_layer(hand.side, mano_root, device)
    n = hand.pose.shape[0]
    pose = torch.from_numpy(hand.pose).to(device)
    trans = torch.from_numpy(hand.trans).to(device)
    betas = torch.from_numpy(hand.betas).to(device)

    verts_out, joints_out = [], []
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out = layer(
                global_orient=pose[s:e, :3],
                hand_pose=pose[s:e, 3:48],
                betas=betas.unsqueeze(0).expand(e - s, -1),
            )
            verts = out.vertices
            joints16 = out.joints  # (B, 16, 3) kinematic joints, no fingertips
            # center on the wrist (manopth center_idx=0); the wrist is the
            # kinematic root, so its posed location is the root joint itself
            wrist = joints16[:, 0:1]
            verts = verts - wrist
            joints16 = joints16 - wrist
            # fingertip joints from mesh vertices (thumb..pinky)
            tips = verts[:, TIP_VERTEX_IDS]
            joints = torch.cat([joints16, tips], dim=1)[:, MANOPTH_JOINT_ORDER]
            # world-frame translation (wrist position, meters)
            verts = verts + trans[s:e].unsqueeze(1)
            joints = joints + trans[s:e].unsqueeze(1)
            verts_out.append(verts.detach().cpu().numpy())
            joints_out.append(joints.detach().cpu().numpy())
    vertices = np.concatenate(verts_out).astype(np.float32)
    joints = np.concatenate(joints_out).astype(np.float32)
    faces = layer.faces.astype(np.int32)
    return vertices, joints, faces
