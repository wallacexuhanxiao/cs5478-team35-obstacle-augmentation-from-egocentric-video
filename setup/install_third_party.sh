#!/bin/bash
# Fetch the third-party code and model files the pipeline uses. Nothing fetched here is redistributed by this
# repository; each project keeps its own license (THIRD_PARTY.md).
#   bash setup/install_third_party.sh            -> third_party/{EgoEngine-CoRL2026,spider,lama_root} + XHand assets
# MANO cannot be downloaded automatically (registration required): see setup/ENVIRONMENTS.md.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$REPO/third_party
mkdir -p "$T"

clone() {  # <url> <dir> <commit>
  if [[ ! -d $2/.git ]]; then git clone "$1" "$2"; fi
  git -C "$2" fetch --quiet origin || true
  git -C "$2" checkout --quiet "$3"
  echo "OK $(basename "$2") @ $(git -C "$2" rev-parse --short HEAD)"
}

# 1. EgoEngine (CoRL 2026 release): the Aria test demonstrations (examples/prepared) and the RB-Y1 renderer.
#    Our additions are copied on top (egoengine_overlay/): augmentation scripts, object models, one patched file.
E=$T/EgoEngine-CoRL2026
clone https://github.com/GaTech-RL2/EgoEngine-CoRL2026.git "$E" b1363c2
if git -C "$E" apply --check "$REPO/egoengine_overlay/replay_trajectory_seg_aria.patch" 2>/dev/null; then
  git -C "$E" apply "$REPO/egoengine_overlay/replay_trajectory_seg_aria.patch"
fi
cp -r "$REPO/egoengine_overlay/aug" "$E/"
cp -r "$REPO/egoengine_overlay/visual" "$E/"
echo "OK EgoEngine overlay applied"

# 2. SPIDER: sampling-based retarget repair (and the XHand URDF + meshes used everywhere).
S=$T/spider
clone https://github.com/facebookresearch/spider.git "$S" 4471700
X=$REPO/third_party/egoengine_repro/assets/xhand/spider
mkdir -p "$X/meshes"
cp "$S"/spider/assets/robots/xhand/*.urdf "$S"/spider/assets/robots/xhand/retarget_config.yaml "$X/"
cp -r "$S"/spider/assets/robots/xhand/assets/. "$X/meshes/"   # the URDFs reference meshes/...
echo "OK XHand assets -> $X"

# 3. LaMa inpainting (removes the human's object from the background frames).
L=$T/lama_root
clone https://github.com/advimman/lama.git "$L/third_party/lama" 786f593
if [[ ! -f $L/data/models/lama/big-lama/models/best.ckpt ]]; then
  mkdir -p "$L/data/models/lama"
  curl -L --fail -o "$L/data/models/lama/big-lama.zip" https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
  (cd "$L/data/models/lama" && unzip -q big-lama.zip)
fi
echo "OK LaMa weights -> $L/data/models/lama/big-lama"
echo "Set LAMA_ROOT=$L in env.sh. Remaining manual step: MANO (setup/ENVIRONMENTS.md)."
