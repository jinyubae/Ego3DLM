# Data preprocessing

This folder contains the code that turns parsed Nymeria recordings into the inputs of Ego3DLM
(`datasets/ego3dlm/`). The splits, the sample-id mapping, the scene QA and the obstacle labels are
released as the annotation bundle (see the main README), so they do not have to be regenerated.

```
preprocess/
├── motion/                      # motion representation, three-point tracking, video embeddings, dataset assembly
│   ├── motion_representation.py # 279-D features, recovered joints, three-point poses, skel_offsets
│   ├── video_features.py        # EVA-ViT-g embeddings of the egocentric frames (2 FPS, 1408-D)
│   ├── build_dataset.py         # per-scene outputs -> flat 6-digit sample ids
│   ├── text_pos_tag.py          # word/POS tags of the narrations
│   └── compute_mean_std.py      # mean.npy / std.npy of the motion features
└── scene/                       # 3D scene features (following 3D-LLM)
    ├── step1_mask2former_masks.py
    ├── step2_eva_pixel_features.py
    ├── step3_lift_to_pointcloud.py
    ├── step4_voxelize.py
    ├── step5_head_pose_voxels.py
    ├── run_scene_pipeline.sh    # steps 1-5 for all scenes
    └── patches/eva_vit_dtype.patch
```

## Input: parsed Nymeria scenes

All scripts read a folder of parsed [Nymeria](https://www.projectaria.com/datasets/nymeria/) recordings (one folder
per recording, called *scene* below). Each recording is cut into narrated segments, one per row of the atomic-action
narrations; all data is sampled at 10 FPS and a segment spans 5 s (50 frames).

```
<parsed_dir>/<scene>/
├── body_pose/NNNN.npy          # (50, 23, 3)   Xsens joint positions of segment NNNN (world frame, z-up)
├── head_pose/NNNN.npy          # (50, 4, 4)    T_world_device of the Aria glasses
├── left_wrist_pose/NNNN.npy    # (50, 4, 4)
├── right_wrist_pose/NNNN.npy   # (50, 4, 4)
├── motion_caption/NNNN.txt     # atomic-action narration of the segment
├── video/NNNN/*.jpg            # egocentric RGB frames of the segment
├── video_all/NNNN.jpg          # all egocentric RGB frames of the recording (1408 x 1408)
├── full_body_pose/full_body_pose.npy   # (T, 23, 3) joints of the whole recording (joint 6 = head)
├── pointcloud/scene_pc.npy     # (50000, 3)    MPS semi-dense point cloud (world frame)
└── projection/projections.npy  # {frame: {'indices': (K,), 'pts_2d': (K, 2)}} for every 50th frame:
                                #   indices of the visible points and their (row, col) pixel in video_all/<frame>.jpg
```

## Environments

* `motion/` (except `video_features.py`) runs in the main Ego3DLM environment (`requirements-freeze.txt`) plus
  spaCy `en_core_web_sm` (`python -m spacy download en_core_web_sm`).
* `scene/` and `motion/video_features.py` follow [3D-LLM](https://github.com/UMass-Embodied-AGI/3D-LLM)
  (`three_steps_3d_feature`): detectron2, its `first_step/mask2former` package and its `second_step/lavis` copy.
  Apply the dtype patch to LAVIS, which lets the EVA-ViT-g encoder run in fp16 or fp32:
  ```bash
  cd 3D-LLM && git apply /path/to/Ego3DLM/preprocess/scene/patches/eva_vit_dtype.patch
  ```
  Mask2Former weights: ADE20K panoptic, Swin-L IN21k, 160k (`model_final_e0c58e.pkl`) with the config
  `configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml` of
  [Mask2Former](https://github.com/facebookresearch/Mask2Former). The EVA-ViT-g weights are downloaded by LAVIS.

## 1. Motion representation

```bash
# 279-D features, recovered joints and three-point poses per scene / segment
python -m preprocess.motion.motion_representation --parsed_dir $PARSED --out_dir $WORK/motion_scene \
    --skel_offsets datasets/ego3dlm/skel_offsets.npy

# egocentric video embeddings (every 5th frame -> 10 x 1408 per segment)
python -m preprocess.motion.video_features --parsed_dir $PARSED --out_dir $WORK/video_scene \
    --lavis_root /path/to/3D-LLM/three_steps_3d_feature/second_step

# flat dataset with the released sample ids, then the narration POS tags
python -m preprocess.motion.build_dataset --scene_index datasets/ego3dlm/scene_info_with_scene_index.txt \
    --motion_dir $WORK/motion_scene --video_dir $WORK/video_scene --parsed_dir $PARSED \
    --out_dir datasets/ego3dlm
python -m preprocess.motion.text_pos_tag --text_dir datasets/ego3dlm/texts
```

* The 279-D feature of a segment with T frames has T-1 rows: root linear velocity (3), root 6D rotation (6),
  root 6D angular velocity (6), 22 joint 6D rotations (132) and 22 joint 6D angular velocities (132).
  Every segment is retargeted to a common skeleton (`skel_offsets.npy`), put on the floor, moved to the XZ origin and
  rotated to face +Z at its first frame.
* `three_points/<id>.npy` (T, 3, 4, 4) are the head / left wrist / right wrist poses in the same canonical frame.
* A sample id is a segment whose next segment of the same recording is the prediction target, so the last segment of
  every recording is not a sample. `scene_info_with_scene_index.txt` maps every id to `scene,segment`.
* `mean.npy` / `std.npy` (`compute_mean_std.py`) and `skel_offsets.npy` are included in the annotation bundle.
* Motion tokens (`TOKENS/`) are produced with the Stage-0 PQ-VAE, see the main README.

## 2. 3D scene features

```bash
PARSED_DIR=$PARSED WORK_DIR=$WORK DATASET_DIR=datasets/ego3dlm \
M2F_ROOT=/path/to/3D-LLM/three_steps_3d_feature/first_step \
M2F_CONFIG=/path/to/maskformer2_swin_large_IN21k_384_bs16_160k.yaml M2F_WEIGHTS=/path/to/model_final_e0c58e.pkl \
LAVIS_ROOT=/path/to/3D-LLM/three_steps_3d_feature/second_step GPUS="0 1 2" \
bash preprocess/scene/run_scene_pipeline.sh
```

| Step | Script | Output |
|---|---|---|
| 1 | `step1_mask2former_masks.py` | panoptic masks (people removed) of every projection frame, 512x512 |
| 2 | `step2_eva_pixel_features.py` | per-pixel EVA-ViT-g features: softmax blend of the global and the per-mask embedding (512x512x1408) |
| 3 | `step3_lift_to_pointcloud.py` | every point takes the feature of the frame where it is closest to the ego head; features are de-duplicated into a codebook  |
| 4 | `step4_voxelize.py` | `3d_features_scene/voxelized_voxels/<scene>.npz` (voxel indices, `min_coords`, 0.15 m) and `voxelized_features/<scene>.pt` (mean feature per voxel) |
| 5 | `step5_head_pose_voxels.py` | `global_head_poses/<id>.npz`: head voxel index and rotation of every frame of a sample |


## 3. Spatial-awareness (obstacle) labels

Directional free space (front / left / right: LOW / MID / HIGH), collision state and the best direction of every frame,
computed from the scene voxels and the head poses. They are included in the annotation bundle; to regenerate them:

```bash
python -m preprocess.parse_obstacle --all --out_dir obstacle_labels --workers 8
```
