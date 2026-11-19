# Obstacle-augmented robot demonstrations from one egocentric human video

CS5478 project, Team 35 — pipeline code and a runnable test sample.

From a single human demonstration recorded with Project Aria glasses, the pipeline produces several robot
demonstrations that perform the same task: the original one, and detours around objects that were not in the
recording. Every demonstration is executed and verified as one continuous physics rollout, and comes with an edited
video in which the human is replaced by a robot and the inserted object is visible. The policy-training experiments of
the report are not part of this repository.

```
human video  ──►  1. reproduce with a dexterous robot hand (PhysX)   ──►  original demonstration + video
                  2. place a planning box across the executed path
                  3. re-plan only the free motion around the box, verify the whole task in PhysX
                  4. certify the box against the measured motion, put visible objects inside it
                  5. render the edited video from the PhysX states  ──►  detour demonstrations + videos
```

## What is where

| Report section | Step | Code |
|---|---|---|
| Problem setting, Reproducing the demonstration | segment at contact events into Interaction Anchors and free motion; retarget the Anchors (IK + SPIDER) | `scripts/derive_oracle_schedule.py`, `scripts/time_scale_schedule.py`, `scripts/prepare_oracle_anchors.py`, `scripts/spider_mpc_proposal.py`, `src/multipath/anchors/` |
| | replay → sampling-based MPC → residual PPO repair of each Anchor from the state actually reached | `src/multipath/anchors/{floating,floating_search,mpc,ppo,ppo_model}.py`, `scripts/develop_reached_anchor.py` |
| | free motion planned with cuRobo, fitted to the source timing, arm reachability | `src/multipath/planning/{floating,bridge,timing,arm}.py`, `scripts/develop_connected_oracle.py` |
| | the whole task re-run as one continuous PhysX rollout (acceptance) | `scripts/run_oracle_v0.py`, `src/multipath/execution/`, `src/multipath/validation/` |
| | driver of the steps above | `scripts/run_end_to_end.py` |
| Obstacle design (planning boxes) | occupancy of the executed motion; boxes that cut the transport path and fit their objects | `scripts/occupancy_ceiling.py`, `scripts/place_planning_box.py` |
| Re-planning around the obstacle | nominal path with the box hidden, pushed out of the box by the L1 linear program | `src/multipath/planning/floating.py` (`minimal_detour`), config `configs/floating_xhand_aria_mustard_1_detour.template.json` |
| Box certificate | minimum distance of the measured hand, object and rendered arm to the box | `scripts/box_clearance.py` |
| Visible objects | objects placed inside a certified box, in the way of the original path | `scripts/place_composite_objects.py` |
| Editing the video | measured states → RB-Y1 commands, human's object removed (LaMa), robot + objects rendered | `egoengine_overlay/aug/{make_visual_package,physics_to_retarget,remove_source_tool,export_rby1_arm}.py`, `scripts/render_edited_video.py` |

`pipeline/run_sample.sh` chains all of it for one recording.

## Requirements

* Linux, one NVIDIA GPU with CUDA 12.8 drivers (tested on an H200), EGL for offscreen rendering.
* Five Python environments (Isaac Sim needs its own; SPIDER and EgoEngine pin incompatible versions). Recipes and the
  exact package lists we used: [`setup/ENVIRONMENTS.md`](setup/ENVIRONMENTS.md), `setup/frozen/`.
* The MANO hand model (free registration; cannot be redistributed).

## Installation

```bash
git clone https://github.com/wallacexuhanxiao/cs5478-team35-obstacle-augmentation-from-egocentric-video.git
cd cs5478-team35-obstacle-augmentation-from-egocentric-video
bash setup/install_third_party.sh     # EgoEngine release (+ our overlay), SPIDER, XHand model, LaMa
# build the five environments: setup/ENVIRONMENTS.md
cp env.example.sh env.sh              # then set the interpreter paths and MANO_ROOT in env.sh
```

## Run the test sample

The test sample is the Aria demonstration `mustard_1` shipped with the EgoEngine release (a person picks up a mustard
bottle and puts it down; 3.1 s). Its parameters are in `samples/mustard_1/case.env`.

```bash
bash pipeline/run_sample.sh           # all stages, about 75 min on one H200
bash pipeline/run_sample.sh reproduce render_original    # or any subset, in order
```

| Stage | Output under `work/mustard_1/` | GPU |
|---|---|---|
| `prepare` | `schedule/`, `proposals/` (segments and retargeted Anchors) | – |
| `reproduce` | `reproduce/final/` (PhysX rollout: `actual_trace.json`, `executed_controls.npz`, `pipeline_result.json`) | yes |
| `render_original` | `render/mustard_1_original/mustard_1_original/visual.mp4` (edited video), `arm_model/` (RB-Y1 arm for planning) | yes |
| `boxes` | `occupancy/`, `boxes/` (`chosen/a.json`: tall box, `chosen/o0.json`: low box) | – |
| `detour_a`, `detour_o0` | `detour_<box>/final/` (verified detour rollouts) | yes |
| `compose_a`, `compose_o0` | RB-Y1 arm of the detour (its rendered surface), `certificate_<box>.json`, `composite_<box>/` (object poses inside the box) | yes (EGL) |
| `render_a`, `render_o0` | `render/mustard_1_box_<box>_<object>/.../visual.mp4` (edited videos of the detours) | yes |

`work/mustard_1/SUMMARY.txt` collects, for every rollout, whether it passed, the number of state resets (must be 0),
the final object error and the executed/source duration, and lists the rendered videos.

## Expected results

Our run of `bash pipeline/run_sample.sh` from a fresh checkout of this repository (one H200; reference outputs in
[`samples/mustard_1/expected/`](samples/mustard_1/expected): `results.json`, `SUMMARY.txt`, boxes, certificates,
object poses, and the five edited videos in `videos/`):

| Rollout | PhysX (one continuous rollout, 0 state resets) | Final bottle error | Detour | Box certificate (min. distance of hand / bottle / rendered arm to the box) | Visible objects in the box |
|---|---|---|---|---|---|
| `reproduce` (no object) | passed, 4.65 s = source duration | 1.5 cm / 0.16 rad | – | – | – |
| `detour_a` (tall box 30 × 16 × 31 cm) | passed, 4.65 s | 1.5 cm / 0.16 rad | around: wrist up to 11.6 cm from the reproduction | 3.6 cm (≥ 1 cm required) | ketchup bottle, wine bottle, cup |
| `detour_o0` (low box 14 × 34 × 12 cm) | passed, 4.65 s | 1.5 cm / 0.16 rad | over: wrist raised 12.7 cm, bottle lifted to 19 cm | 3.6 cm | cup (the bottles are too tall for this box) |

Run times on one H200: prepare 5 min (CPU), reproduce 6 min, render_original 5 min, boxes 1 min, detour_a 10 min,
detour_o0 13 min, compose 2–3 min per box, 4 min per rendered video; about 75 min in total. PhysX, cuRobo and the
PPO/MPC repairs involve sampling, so a rerun may pick other repair candidates or box poses and give slightly different
numbers; acceptance is always decided by the final continuous rollout.

## Notes

* The physics robot is a floating XHand (a 6-DoF wrist driven by an external wrench plus 12 actuated finger joints);
  the RB-Y1 arm is attached only kinematically (reachability during planning, rendering).
* The source clock is slowed uniformly by 1.5 (as in the EgoEngine release, which plays the Aria data at 20 Hz).
* The planning box exists only for the planner: it is neither in the physics nor in the video. The visible objects are
  never simulated; the box certificate guarantees that the executed motion stays clear of every object inside the box.
* The report's "mug" is a TACO mesh whose licence does not allow redistribution; the test sample uses a generated cup
  of the same height instead.
* Third-party code and licences: [`THIRD_PARTY.md`](THIRD_PARTY.md).
