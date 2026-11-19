# Copy to env.sh (git-ignored) and edit the paths. Sourced by every script under pipeline/.
# See setup/ENVIRONMENTS.md for how each interpreter was built (exact package lists in setup/frozen/).

export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export WORK=${WORK:-$REPO/work}                       # every output of the pipeline goes here

# Third-party checkouts (setup/install_third_party.sh creates them at these default places).
export EGOENGINE_DIR=${EGOENGINE_DIR:-$REPO/third_party/EgoEngine-CoRL2026}  # official release @ b1363c2 + egoengine_overlay/
export SPIDER_DIR=${SPIDER_DIR:-$REPO/third_party/spider}                    # facebookresearch/spider @ 4471700
export LAMA_ROOT=${LAMA_ROOT:-$REPO/third_party/lama_root}                   # advimman/lama @ 786f593 + big-lama weights
export MANO_ROOT=${MANO_ROOT:-/path/to/mano_smplx}    # contains mano/MANO_RIGHT.pkl and mano/MANO_LEFT.pkl (setup/ENVIRONMENTS.md)

# Python interpreters (one virtual environment each; setup/ENVIRONMENTS.md).
export PY_BASE=${PY_BASE:-/path/to/envs/base/bin/python}             # schedule, retargeting, geometry, LaMa
export PY_SIM=${PY_SIM:-/path/to/envs/sim/bin/python}                # Isaac Sim 5.1 + Isaac Lab 2.3.2 + cuRobo 0.7.7
export PY_SPIDER=${PY_SPIDER:-/path/to/envs/spider/bin/python}       # SPIDER + MuJoCo Warp (sampling-based retarget repair)
export PY_EGO_ACTION=${PY_EGO_ACTION:-/path/to/envs/ego_action/bin/python}  # EgoEngine package builder
export PY_EGO_VISUAL=${PY_EGO_VISUAL:-/path/to/envs/ego_visual/bin/python}  # EgoEngine RB-Y1 renderer (MuJoCo EGL)

# Isaac Sim processes are started as: [SIM_CONTAINER...] env PYTHONPATH=... <vars> $PY_SIM -u <script> ...
# Leave SIM_CONTAINER empty on a workstation. On a cluster that runs Isaac inside a container, give the container
# command up to the image, e.g.  SIM_CONTAINER="singularity exec --nv -e /path/to/image.sif"
export SIM_CONTAINER=${SIM_CONTAINER:-}
export SIM_PYTHONPATH_EXTRA=${SIM_PYTHONPATH_EXTRA:-}  # extra site dir of the Isaac env (e.g. cuRobo installed with --target)
export SIM_EXTRA_ENV=${SIM_EXTRA_ENV:-}                # extra VAR=value words for the Isaac processes (e.g. LD_LIBRARY_PATH=...)

export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$WORK/.warp_cache}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
