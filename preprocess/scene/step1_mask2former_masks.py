"""Step 1 - Mask2Former panoptic masks for the projection frames of a scene (3D-LLM, first step).

For every frame key in ``<parsed_dir>/<scene>/projection/projections.npy`` the egocentric frame
``video_all/NNNN.jpg`` is segmented with Mask2Former (Swin-L, ADE20K panoptic). The masks of all segments
except people are resized to 512x512 and stored as

    <out_dir>/<scene>/NNNN.pt   (num_masks, 512, 512) float16

Requires detectron2 and the ``mask2former`` package (e.g. ``3D-LLM/three_steps_3d_feature/first_step``),
whose parent folder is passed through ``--mask2former_root``.
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
from tqdm import tqdm

LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512
ADE20K_PERSON_ID = 12


def setup_cfg(args):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def masks_without_person(panoptic_map, segments_info):
    """Binary masks (num_masks, H, W) of all panoptic segments except people; None if there are none."""
    seg_ids = [seg['id'] for seg in segments_info]
    if not seg_ids:
        return None
    assert len(torch.Tensor(seg_ids).unique()) == len(seg_ids)
    assert min(seg_ids) == 1

    mask_list = []
    for seg_id in seg_ids:
        if seg_id == 0:
            continue
        if segments_info[seg_id - 1]['category_id'] == ADE20K_PERSON_ID:
            continue
        mask_list.append(panoptic_map == seg_id)
    return torch.stack(mask_list) if mask_list else None


def main():
    parser = argparse.ArgumentParser(description="Mask2Former panoptic masks for the projection frames")
    parser.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes")
    parser.add_argument("--scene", required=True, help="scene name")
    parser.add_argument("--out_dir", required=True, help="masks are written to <out_dir>/<scene>/NNNN.pt")
    parser.add_argument("--mask2former_root", required=True, help="folder that contains the `mask2former` package")
    parser.add_argument("--config_file", required=True,
                        help="maskformer2_swin_large_IN21k_384_bs16_160k.yaml (ADE20K panoptic)")
    parser.add_argument("--weights", required=True, help="model_final_e0c58e.pkl")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="subset of projection frame keys")
    args = parser.parse_args()

    sys.path.insert(0, args.mask2former_root)
    from detectron2.data.detection_utils import read_image
    from detectron2.engine.defaults import DefaultPredictor

    torch.autograd.set_grad_enabled(False)
    cfg = setup_cfg(types.SimpleNamespace(config_file=args.config_file, opts=["MODEL.WEIGHTS", args.weights]))
    predictor = DefaultPredictor(cfg)

    scene_dir = os.path.join(args.parsed_dir, args.scene)
    save_dir = os.path.join(args.out_dir, args.scene)
    os.makedirs(save_dir, exist_ok=True)

    projected = np.load(os.path.join(scene_dir, 'projection', 'projections.npy'), allow_pickle=True).item()
    frames = sorted(projected.keys())
    if args.frames is not None:
        frames = [f for f in frames if f in set(args.frames)]

    for frame in tqdm(frames, desc=args.scene):
        global_frame = frame + int(projected[frame].get("sub_frame", 0))
        img = read_image(os.path.join(scene_dir, 'video_all', f"{global_frame:04d}.jpg"), format="BGR")
        predictions = predictor(img)
        masks = masks_without_person(predictions['panoptic_seg'][0], predictions['panoptic_seg'][1])
        if masks is None:
            continue
        masks = torch.nn.functional.interpolate(masks.float().unsqueeze(0), [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH],
                                                mode="nearest").half()
        torch.save(masks[0].detach().cpu(), os.path.join(save_dir, f"{frame:04d}.pt"))


if __name__ == "__main__":
    main()
