# Environments

The pipeline calls five Python interpreters, because its parts come from projects with incompatible pins
(Isaac Sim needs Python 3.11 and NumPy 1.26; SPIDER and the EgoEngine action code need Python 3.12; the EgoEngine
renderer is tested on Python 3.10). Each one is an ordinary virtual environment; set its interpreter in `env.sh`.
The exact package lists of the environments used for the report are in `setup/frozen/*.txt`.

Hardware: Linux, one NVIDIA GPU with CUDA 12.8 drivers (tested on H200; any RTX-class GPU supported by Isaac Sim 5.1
should work), EGL for offscreen rendering, ~40 GB disk for the environments.
We used [uv](https://docs.astral.sh/uv/) (`pip install uv`); plain `pip` works the same way.

| `env.sh` variable | Python | Used for | Reference list |
|---|---|---|---|
| `PY_BASE` | 3.11 | schedule, retargeting (mink/MuJoCo), planning boxes, certificates, LaMa inpainting | `frozen/base.txt` |
| `PY_SIM` | 3.11 | Isaac Sim 5.1 / Isaac Lab 2.3.2 PhysX execution, cuRobo planning, MPC, residual PPO | `frozen/sim.txt` |
| `PY_SPIDER` | 3.12 | SPIDER sampling-based retarget repair (MuJoCo Warp) | `frozen/spider.txt` |
| `PY_EGO_ACTION` | 3.12 | EgoEngine package builder (`aug/make_visual_package.py`) | `frozen/ego_action.txt` |
| `PY_EGO_VISUAL` | 3.10 | EgoEngine RB-Y1 renderer and compositor | `frozen/ego_visual.txt` |

Run `bash setup/install_third_party.sh` first (it clones EgoEngine, SPIDER and LaMa into `third_party/`).

## `PY_BASE`

```bash
uv venv --python 3.11 envs/base
uv pip install --python envs/base/bin/python "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/base/bin/python -r setup/requirements-base.txt
```

## `PY_SIM` (Isaac Sim + cuRobo)

```bash
uv venv --python 3.11 envs/sim
uv pip install --python envs/sim/bin/python "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/sim/bin/python "isaaclab[all]==2.3.2" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
# cuRobo v0.7.7 from source (needs the CUDA toolkit; set TORCH_CUDA_ARCH_LIST to your GPU, e.g. 8.9 or 9.0)
git clone https://github.com/NVlabs/curobo.git third_party/curobo && git -C third_party/curobo checkout v0.7.7
uv pip install --python envs/sim/bin/python --no-build-isolation ./third_party/curobo
uv pip install --python envs/sim/bin/python pycollada==0.9.3 python-fcl==0.7.0.8 numpy-quaternion==2024.0.13 \
  daqp==0.9.1 lxml==6.1.3 scikit-image==0.26.0 "trimesh==4.5.1"
OMNI_KIT_ACCEPT_EULA=YES envs/sim/bin/python -c "import isaacsim, curobo; print('ok')"
```

Isaac Sim asks to accept its EULA on first start; the pipeline passes `OMNI_KIT_ACCEPT_EULA=YES`.
On a cluster where Isaac must run in a container, set `SIM_CONTAINER` (and, if needed, `SIM_EXTRA_ENV`,
`SIM_PYTHONPATH_EXTRA`) in `env.sh`; the pipeline then starts every Isaac process as
`$SIM_CONTAINER env PYTHONPATH=... $PY_SIM -u <script>`.

## `PY_SPIDER`

```bash
uv venv --python 3.12 envs/spider
uv pip install --python envs/spider/bin/python "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/spider/bin/python --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
  "mujoco==3.7.0" "mujoco-warp==3.7.0.1" "warp-lang==1.12.1" numpy scipy loguru tyro hydra-core omegaconf \
  "imageio[ffmpeg]" mediapy matplotlib opencv-python-headless "rerun-sdk==0.26.2" loop-rate-limiters trimesh coacd
```

## `PY_EGO_ACTION` and `PY_EGO_VISUAL` (EgoEngine)

Follow the EgoEngine README, with two fixes we needed:

```bash
E=third_party/EgoEngine-CoRL2026
uv venv --python 3.12 envs/ego_action
uv pip install --python envs/ego_action/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/ego_action/bin/python --no-sources -r $E/requirements/action-tested.txt   # --no-sources: see below
uv pip install --python envs/ego_action/bin/python --no-sources -e $E
uv pip install --python envs/ego_action/bin/python "warp-lang==1.11.1" tensorboard

uv venv --python 3.10 envs/ego_visual
uv pip install --python envs/ego_visual/bin/python -r $E/requirements/visual-tested.txt
uv pip install --python envs/ego_visual/bin/python --no-deps -e $E
```

* `--no-sources`: `mujoco_warp` declares a `tool.uv.sources` entry that points `mujoco` to a nightly index.
* `warp-lang==1.11.1`: the pinned `mujoco_warp` revision uses `wp.math.sqrt`, which newer Warp releases reject.

## MANO (manual)

The Aria demonstrations store hand keypoints; turning them into hand poses uses the MANO model, whose licence does
not allow redistribution. Register at <https://mano.is.tue.mpg.de>, download "Models & Code", and place the files as

```
$MANO_ROOT/mano/MANO_RIGHT.pkl
$MANO_ROOT/mano/MANO_LEFT.pkl
```
