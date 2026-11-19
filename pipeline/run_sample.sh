#!/bin/bash
# Full pipeline on one egocentric demonstration (default: the test sample samples/mustard_1).
#
#   bash pipeline/run_sample.sh [stage ...]        stages run in the order given; default: all
#
#   prepare           segment the demonstration (Anchors / free motion), slow the clock, retarget each Anchor
#   reproduce         reproduce the demonstration with the floating XHand: replay -> MPC -> residual PPO repairs,
#                     cuRobo free motion, one continuous PhysX rollout                    [Isaac Sim, GPU]
#   render_original   edited video of the reproduction (human removed, RB-Y1 rendered from the PhysX states);
#                     exports the rendered arm surface and the RB-Y1 arm model used for planning   [GPU, EGL]
#   boxes             occupancy map of the executed motion, planning boxes across the transport path
#                     (two tall "go around" boxes, one low "go over" box)
#   detour_a          re-plan around box a (minimal-deviation LP), verify as one continuous PhysX rollout  [GPU]
#   detour_o0         same for the low box o0                                            [GPU]
#   compose_a         box certificate on the measured rollout + visible objects placed inside box a
#   compose_o0        same for box o0
#   render_a          edited videos of detour a, one per placed object                   [GPU, EGL]
#   render_o0         same for detour o0                                                  [GPU, EGL]
#
# Requires env.sh (copy env.example.sh). Outputs: $WORK/<case>/... ; summary: $WORK/<case>/SUMMARY.txt
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$REPO/env.sh" ]] || { echo "missing $REPO/env.sh (copy env.example.sh and edit it)"; exit 2; }
source "$REPO/env.sh"
CASE_ENV=${CASE_ENV:-$REPO/samples/mustard_1/case.env}
source "$CASE_ENV"
W=$WORK/$CASE
mkdir -p "$W/logs"
cd "$REPO"
export PYTHONPATH=$REPO/src:$REPO/scripts:$REPO/third_party/egoengine_repro/src
export MANO_ROOT PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
URDF=$REPO/third_party/egoengine_repro/assets/xhand/spider/xhand_right.urdf
SCHED=$W/schedule/schedule_x${TIME_SCALE/./p}.json

say() { echo "[$(date +%H:%M:%S)] $CASE $*" | tee -a "$W/SUMMARY.txt"; }
base() { "$PY_BASE" -u "$@"; }

sim_prefix() {  # <json out> [VAR=value ...]: argv prefix of every Isaac Sim process of one run
  local out=$1; shift
  "$PY_BASE" - "$out" $SIM_CONTAINER env "PYTHONPATH=$REPO/src${SIM_PYTHONPATH_EXTRA:+:$SIM_PYTHONPATH_EXTRA}" \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "XDG_CACHE_HOME=$(dirname "$out")/cache" \
    "OMP_NUM_THREADS=$OMP_NUM_THREADS" $SIM_EXTRA_ENV "$@" "$PY_SIM" -u <<'PY'
import json, sys
json.dump(sys.argv[2:], open(sys.argv[1], 'w'))
PY
}

end_to_end() {  # <run dir> <floating config> [environment objects json] : reproduce / detour as one verified rollout
  local out=$1 cfg=$2 env_objects=${3:-}
  mkdir -p "$out/cache"
  if [[ -n "$env_objects" ]]; then
    sim_prefix "$out/sim_prefix.json" "MULTIPATH_ENVIRONMENT_OBJECTS=$env_objects"
  else
    sim_prefix "$out/sim_prefix.json"
  fi
  MULTIPATH_ENVIRONMENT_OBJECTS=$env_objects WARP_CACHE_PATH=$WARP_CACHE_PATH base scripts/run_end_to_end.py \
    --schedule "$SCHED" --triplet aria-mustard --sequence "$CASE" --mount-report configs/franka_xhand_mount.json \
    --floating-config "$cfg" --xhand-urdf "$URDF" --object-cache "$W/object_cache" --sim-prefix "$out/sim_prefix.json" \
    --output-dir "$out" --prepared-proposals "$W/proposals" --retarget egoengine_wrap \
    --spider-python "$PY_SPIDER" --spider-dir "$SPIDER_DIR" --spider-overrides "$REPO/configs/spider_xhand_mpc.json" \
    --mpc-candidates 16 --mpc-iters 2 --ppo-updates 40 --max-attempts 6 --solutions-per-interval 1 > "$out/run.log" 2>&1 || true
  base - "$out" <<'PY' | tee -a "$W/SUMMARY.txt"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
r = json.loads((out / 'final' / 'pipeline_result.json').read_text()) if (out / 'final' / 'pipeline_result.json').exists() else {}
t = r.get('task', {})
ok = bool(r.get('full_pipeline_success') and r.get('task_success'))
print(f'  {out.name}: physics {"PASSED" if ok else "FAILED"}; continuous rollout, state resets between segments: '
      f'{r.get("state_resets_between_segments")}; final object error {t.get("final_errors")}; '
      f'executed/source duration {r.get("actual_duration_s")}/{r.get("source_duration_s")} s')
sys.exit(0 if ok else 1)
PY
}

egl_device() { (cd "$EGOENGINE_DIR" && "$PY_EGO_VISUAL" "$REPO/pipeline/egl_probe.py" 2>/dev/null || echo 0); }

robot_commands() {  # <name> <run dir> [composite json]: visual package + RB-Y1 commands from the measured PhysX states
  # (inverse kinematics of the arm to the measured wrist); also exports the rendered arm surface arm_surface_sim.npz
  local name=$1 run=$2 composite=${3:-} pkg=$W/render/pkg_$1 log=$W/logs/render_$1
  rm -rf "$pkg"; mkdir -p "$W/render" "$log"
  (cd "$EGOENGINE_DIR" && PYTHONPATH=$EGOENGINE_DIR "$PY_EGO_ACTION" aug/make_visual_package.py --source-root examples/prepared --case "$CASE" \
     --name "$name" --out "$pkg" > "$log/package.log" 2>&1)
  (cd "$EGOENGINE_DIR" && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$(egl_device) PYTHONPATH=$EGOENGINE_DIR:$REPO/src \
     "$PY_EGO_VISUAL" aug/physics_to_retarget.py --package "$pkg" --case "$name" --source-root "$EGOENGINE_DIR/examples/prepared" \
     --multipath-run "$run/final" --multipath-manifest "$run/proposals/manifest.json" --source-case "$CASE" \
     --object-cache "$W/object_cache" ${composite:+--composite-object "$composite"} > "$log/physics_to_retarget.log" 2>&1)
}

render() {  # <name> <run dir> [composite json] [obstacle xml]: edited video from the measured PhysX states
  local name=$1 run=$2 composite=${3:-} xml=${4:-} pkg=$W/render/pkg_$1 log=$W/logs/render_$1
  robot_commands "$name" "$run" "$composite"
  local egl; egl=$(egl_device)
  local vi=$pkg/visual_inputs/Aria/$name
  (cd "$EGOENGINE_DIR" && base aug/remove_source_tool.py --visual-dir "$vi" --source-retarget "$vi/retarget_source.json" \
     --mesh "$pkg/assets/95ab61c37240_mustard.obj" --lama-root "$LAMA_ROOT" --helpers "$REPO/scripts" > "$log/remove_source_tool.log" 2>&1)
  (cd "$EGOENGINE_DIR" && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$egl EGOENGINE_OBJECTS_FOREGROUND=1 EGOENGINE_OBSTACLE_XML=$xml \
     PYTHONPATH=$EGOENGINE_DIR "$PY_EGO_VISUAL" -m egoengine --data-root "$pkg" demo --case "$name" --output "$W/render/$name" \
     > "$log/render.log" 2>&1)
  say "  video $name: work/$CASE/render/$name/$name/visual.mp4 (edited video: RB-Y1 + objects over the real, human-free background); render/scene.mp4 (same motion in the simulator scene)"
}

stage_prepare() {
  say "prepare: segment, slow the clock x$TIME_SCALE, retarget the Anchors"
  mkdir -p "$W/schedule" "$W/object_cache"
  base scripts/derive_oracle_schedule.py --triplet aria-mustard --sequence "$CASE" --aria-release "$EGOENGINE_DIR" \
    --aria-case "$CASE" --object-cache "$W/object_cache" --output "$W/schedule/schedule.json" --variant "$SCHEDULE_VARIANT" \
    --quiet-boundaries > "$W/logs/prepare.log" 2>&1
  base scripts/time_scale_schedule.py --schedule "$W/schedule/schedule.json" --scale "$TIME_SCALE" --output "$SCHED" >> "$W/logs/prepare.log" 2>&1
  base scripts/prepare_oracle_anchors.py --schedule "$SCHED" --triplet aria-mustard --sequence "$CASE" \
    --mount-report configs/franka_xhand_mount.json --object-cache "$W/object_cache" --output-dir "$W/proposals" \
    --retarget egoengine_wrap --aria-release "$EGOENGINE_DIR" --aria-case "$CASE" >> "$W/logs/prepare.log" 2>&1
  say "  $(grep -h -E '^SCHEDULE' "$W/logs/prepare.log" | cut -c1-160)"
}

stage_reproduce() {
  say "reproduce: no inserted object"
  end_to_end "$W/reproduce" "$REPO/$FLOATING_CONFIG"
}

stage_render_original() {
  say "render_original"
  render "${CASE}_original" "$W/reproduce"
  local vi=$W/render/pkg_${CASE}_original/visual_inputs/Aria/${CASE}_original
  (cd "$EGOENGINE_DIR" && MUJOCO_GL=egl PYTHONPATH=$EGOENGINE_DIR "$PY_EGO_VISUAL" -u aug/export_rby1_arm.py --visual-dir "$vi" \
     --manifest "$W/reproduce/proposals/manifest.json" --hand-offset "$vi/physics_to_retarget.json" --cell 0.02 \
     --out "$W/arm_model" > "$W/logs/export_arm.log" 2>&1)
  sed "s|@ARM_MODEL@|$W/arm_model/arm_model.json|" "$REPO/$DETOUR_TEMPLATE" > "$W/floating_config_detour.json"
  say "  RB-Y1 arm model: $W/arm_model/arm_model.json"
}

stage_boxes() {
  say "boxes: occupancy of the executed motion, planning boxes across the transport"
  local arm=$W/render/pkg_${CASE}_original/visual_inputs/Aria/${CASE}_original/arm_surface_sim.npz
  local ef; ef=$(base -c "import json;print(json.load(open('$SCHED'))['windows'][0]['end'])")
  base scripts/occupancy_ceiling.py --run-dir "$W/reproduce/final" --schedule "$SCHED" --window 2 --extra-free 1 "$ef" \
    --free-tool --keep-surface "$arm" --table-xy $TABLE_XY --objects $OCC_OBJECTS --out "$W/occupancy" > "$W/logs/boxes.log" 2>&1
  local common=(--run-dir "$W/reproduce/final" --schedule "$SCHED" --window 2 --block-match right_hand /World/tool
                --table-xy $TABLE_XY --occupancy "$W/occupancy" --step 0.02 --depth 0.01 --end-clearance 0.02)
  base scripts/place_planning_box.py "${common[@]}" --pocket-objects $TALL_POCKET --sizes $TALL_SIZES --yaws 4 --count 2 \
    --min-separation 0.05 --out-dir "$W/boxes/tall" >> "$W/logs/boxes.log" 2>&1
  base scripts/place_planning_box.py "${common[@]}" --pocket-objects $OVER_POCKET --box-height "$OVER_HEIGHT" --sizes $OVER_SIZES \
    --yaws 8 --count 1 --out-dir "$W/boxes/over" >> "$W/logs/boxes.log" 2>&1
  mkdir -p "$W/boxes/chosen"
  cp "$W/boxes/tall/box_0.json" "$W/boxes/chosen/a.json"
  cp "$W/boxes/over/box_0.json" "$W/boxes/chosen/o0.json"
  grep -h -E '^BOX' "$W/logs/boxes.log" | cut -c1-200 | while read -r l; do say "  $l"; done
}

stage_detour() {  # <box>
  say "detour_$1: re-plan around planning box $1 (planner only; not in physics or video)"
  end_to_end "$W/detour_$1" "$W/floating_config_detour.json" "$W/boxes/chosen/$1.json"
}

stage_compose() {  # <box>
  local k=$1
  say "compose_$k: box certificate on the measured rollout, visible objects inside the box"
  # the RB-Y1 arm that will be rendered for THIS rollout (it is not in the physics, so it is certified separately)
  robot_commands "${CASE}_box_${k}" "$W/detour_$k"
  local arm=$W/render/pkg_${CASE}_box_${k}/visual_inputs/Aria/${CASE}_box_${k}/arm_surface_sim.npz
  base scripts/box_clearance.py --run-dir "$W/detour_$k/final" --box "$W/boxes/chosen/$k.json" --keep-surface "$arm" \
    > "$W/certificate_$k.json"
  say "  certificate: $(cut -c1-200 "$W/certificate_$k.json")"
  local mustard; mustard=$(ls "$W"/object_cache/*mustard*/*_m.obj | head -1)
  local q; q=$(base -c "import numpy as np;from scipy.spatial.transform import Rotation as R;q=R.from_matrix(np.load('$REPO/samples/$CASE/assets/mustard_rest_rotation.npy')).as_quat();print(','.join(map(str,q[[3,0,1,2]])))")
  rm -rf "$W/composite_$k"; mkdir -p "$W/composite_$k"
  base scripts/place_composite_objects.py --certificate "$W/certificate_$k.json" --run-dir "$W/detour_$k/final" \
    --box "$W/boxes/chosen/$k.json" --nominal-run "$W/reproduce/final" --schedule "$SCHED" --window 2 --table-xy $TABLE_XY \
    --keep-surface "$arm" --object "ketchup=$mustard=$q=8" \
    --object "wine_bottle=$REPO/samples/$CASE/assets/wine_bottle_m.obj=1,0,0,0=1" \
    --object "cup=$REPO/samples/$CASE/assets/cup_m.obj=1,0,0,0=16" --variants 1 --out-dir "$W/composite_$k" \
    > "$W/logs/compose_$k.log" 2>&1
  grep -h -E 'COMPOSITE' "$W/logs/compose_$k.log" | cut -c1-200 | while read -r l; do say "  $l"; done
}

stage_render_detour() {  # <box>
  local k=$1
  say "render_$k"
  for f in "$W/composite_$k"/*.json; do
    [[ -f $f ]] || continue
    local stem; stem=$(basename "$f" .json)
    render "${CASE}_box_${k}_$stem" "$W/detour_$k" "$f" "${stem%_[0-9]}_visual.xml"
  done
}

STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(prepare reproduce render_original boxes detour_a detour_o0 compose_a compose_o0 render_a render_o0)
for s in "${STAGES[@]}"; do
  case $s in
    prepare|reproduce|render_original|boxes) "stage_$s" ;;
    detour_*) stage_detour "${s#detour_}" ;;
    compose_*) stage_compose "${s#compose_}" ;;
    render_*) stage_render_detour "${s#render_}" ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
done
say "done: ${STAGES[*]}"
