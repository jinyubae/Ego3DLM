#!/usr/bin/env bash
# 3D scene features for Ego3DLM (steps 1-4 per scene, distributed over GPUs; step 5 once at the end).
#
# Required environment variables:
#   PARSED_DIR    root of the parsed Nymeria scenes (<scene>/{video_all,projection,pointcloud,full_body_pose,head_pose})
#   WORK_DIR      scratch space for the intermediate masks / features / lifted point clouds
#   DATASET_DIR   dataset root, e.g. datasets/ego3dlm
#   M2F_ROOT      folder containing the `mask2former` package (3D-LLM/three_steps_3d_feature/first_step)
#   M2F_CONFIG    maskformer2_swin_large_IN21k_384_bs16_160k.yaml (ADE20K panoptic)
#   M2F_WEIGHTS   model_final_e0c58e.pkl
#   LAVIS_ROOT    folder containing the patched `lavis` package (3D-LLM/three_steps_3d_feature/second_step)
# Optional:
#   GPUS          space separated GPU ids (default: "0")
#   SCENE_INDEX   id,scene,segment mapping (default: $DATASET_DIR/scene_info_with_scene_index.txt)
set -euo pipefail

: "${PARSED_DIR:?}" "${WORK_DIR:?}" "${DATASET_DIR:?}" "${M2F_ROOT:?}" "${M2F_CONFIG:?}" "${M2F_WEIGHTS:?}" "${LAVIS_ROOT:?}"
GPUS=(${GPUS:-0})
SCENE_INDEX=${SCENE_INDEX:-$DATASET_DIR/scene_info_with_scene_index.txt}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MASK_DIR=$WORK_DIR/mask
FEAT_DIR=$WORK_DIR/pixel_feat
PROJ_DIR=$WORK_DIR/proj
SCENE_OUT=$DATASET_DIR/3d_features_scene

process_one_scene() {
  local scene="$1" gpu="$2"
  if [[ -f "$SCENE_OUT/voxelized_voxels/$scene.npz" ]]; then
    echo "[skip][GPU $gpu] $scene"
    return 0
  fi
  echo "[start][GPU $gpu] $scene"
  (
    cd "$REPO_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u preprocess/scene/step1_mask2former_masks.py --parsed_dir "$PARSED_DIR" --scene "$scene" \
        --out_dir "$MASK_DIR" --mask2former_root "$M2F_ROOT" --config_file "$M2F_CONFIG" --weights "$M2F_WEIGHTS"
    python -u preprocess/scene/step2_eva_pixel_features.py --parsed_dir "$PARSED_DIR" --scene "$scene" \
        --mask_dir "$MASK_DIR" --out_dir "$FEAT_DIR" --lavis_root "$LAVIS_ROOT"
    python -u preprocess/scene/step3_lift_to_pointcloud.py --parsed_dir "$PARSED_DIR" --scene "$scene" \
        --feat_dir "$FEAT_DIR" --out_dir "$PROJ_DIR"
    python -u preprocess/scene/step4_voxelize.py --parsed_dir "$PARSED_DIR" --proj_dir "$PROJ_DIR" --scene "$scene" \
        --out_dir "$SCENE_OUT"
  )
  # the dense per-pixel features are large (~0.7 GB per frame); drop them once the scene is lifted
  rm -rf -- "${FEAT_DIR:?}/$scene"
  echo "[done][GPU $gpu] $scene"
}

worker() {
  local gpu="$1"; shift
  for scene in "$@"; do
    process_one_scene "$scene" "$gpu"
  done
}

mapfile -t scenes < <(find "$PARSED_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
num_gpus=${#GPUS[@]}
pids=()
for ((g = 0; g < num_gpus; g++)); do
  part=()
  for ((i = g; i < ${#scenes[@]}; i += num_gpus)); do
    part+=("${scenes[i]}")
  done
  worker "${GPUS[g]}" "${part[@]}" &
  pids+=($!)
done
wait "${pids[@]}"

# step 5: head poses of every sample in the voxel grid of its scene
cd "$REPO_DIR"
python -u preprocess/scene/step5_head_pose_voxels.py --parsed_dir "$PARSED_DIR" --scene_index "$SCENE_INDEX" \
    --voxel_dir "$SCENE_OUT/voxelized_voxels" --out_dir "$DATASET_DIR/global_head_poses"
echo "All done."
