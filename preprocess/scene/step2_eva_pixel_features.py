"""Step 2 - dense per-pixel EVA-ViT-g features from the Mask2Former masks (3D-LLM, second step).

For every mask file of step 1, the frame is encoded globally and per mask (each mask region pasted on a
blank canvas). Every pixel of a mask receives a softmax-weighted blend of the global and the mask feature:

    <out_dir>/<scene>/NNNN.pt   (512, 512, 1408) float16

Requires the LAVIS copy shipped with 3D-LLM (``three_steps_3d_feature/second_step``) with
``patches/eva_vit_dtype.patch`` applied, passed through ``--lavis_root``.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512


def get_bbox_around_mask(mask):
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        bbox = (0, 0, mask.shape[0], mask.shape[1])
    else:
        topleft = nonzero_inds.min(0)[0]
        botright = nonzero_inds.max(0)[0]
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())
    return bbox, nonzero_inds


@torch.no_grad()
def encode(visual_encoder, image, device):
    feat = visual_encoder(image).detach().clone().half().to(device)
    return torch.nn.functional.normalize(feat.mean(1), dim=-1)  # (1, 1408)


def pixel_features(raw_image, mask, visual_encoder, device):
    image = torch.tensor(raw_image[:512, :512]).permute(2, 0, 1).unsqueeze(0).to(device)
    global_feat = encode(visual_encoder, image, device)
    feat_dim = global_feat.shape[-1]
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    outfeat = torch.zeros(512, 512, feat_dim, dtype=global_feat.dtype)
    mask = mask.unsqueeze(0)[:, :, :512, :512]  # 1, num_masks, H, W
    num_masks = mask.shape[-3]

    rois, roi_sims, feat_per_roi, roi_nonzero_inds = [], [], [], []
    for _i in range(num_masks):
        bbox, nonzero_inds = get_bbox_around_mask(mask[0, _i])
        x0, y0, x1, y1 = bbox
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if bbox_area / (LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT) < 0.005:
            continue

        roi = torch.ones((512, 512, 3))
        roi[x0:x1, y0:y1] = torch.tensor(raw_image[:512, :512])[x0:x1, y0:y1]
        roifeat = encode(visual_encoder, roi.permute(2, 0, 1).unsqueeze(0).to(device), device)
        feat_per_roi.append(roifeat)
        roi_nonzero_inds.append(nonzero_inds)
        rois.append(torch.tensor(list(bbox)))
        roi_sims.append(cosine_similarity(global_feat, roifeat))

    if len(rois) == 0:
        return None
    rois = torch.stack(rois)
    scores = torch.cat(roi_sims).to(rois.device)
    retained = torchvision.ops.nms(rois.float().cpu(), scores.float().cpu(), iou_threshold=1.0)
    feat_per_roi = torch.cat(feat_per_roi, dim=0)

    retained_scores = scores[retained]
    retained_feat = feat_per_roi[retained]
    retained_nonzero_inds = [roi_nonzero_inds[retained[k].item()] for k in range(retained.shape[0])]

    # down-weight masks that are similar to all other masks
    mask_sim_mat = torch.nn.functional.cosine_similarity(retained_feat[:, :, None], retained_feat.t()[None, :, :])
    mask_sim_mat.fill_diagonal_(0.0)
    mask_sim_mat = mask_sim_mat.mean(1)
    softmax_scores = torch.nn.functional.softmax(retained_scores.to(device) - mask_sim_mat, dim=0)

    for k in range(retained.shape[0]):
        weighted = softmax_scores[k] * global_feat + (1 - softmax_scores[k]) * retained_feat[k]
        weighted = torch.nn.functional.normalize(weighted, dim=-1)
        rows, cols = retained_nonzero_inds[k][:, 0], retained_nonzero_inds[k][:, 1]
        outfeat[rows, cols] += weighted[0].detach().cpu().half()
        outfeat[rows, cols] = torch.nn.functional.normalize(outfeat[rows, cols].float(), dim=-1).half()

    outfeat = outfeat.unsqueeze(0).float().permute(0, 3, 1, 2)  # 1, D, H, W
    outfeat = torch.nn.functional.interpolate(outfeat, [512, 512], mode="nearest")
    outfeat = torch.nn.functional.normalize(outfeat.permute(0, 2, 3, 1), dim=-1)
    return outfeat[0].half()  # H, W, D


def main():
    parser = argparse.ArgumentParser(description="Per-pixel EVA-ViT-g features from Mask2Former masks")
    parser.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes")
    parser.add_argument("--scene", required=True, help="scene name")
    parser.add_argument("--mask_dir", required=True, help="output root of step 1")
    parser.add_argument("--out_dir", required=True, help="features are written to <out_dir>/<scene>/NNNN.pt")
    parser.add_argument("--lavis_root", required=True, help="folder that contains the (patched) `lavis` package")
    args = parser.parse_args()

    sys.path.insert(0, args.lavis_root)
    from lavis.models.eva_vit import create_eva_vit_g

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = create_eva_vit_g(512).to(device).half()
    visual_encoder.eval()

    scene_dir = os.path.join(args.parsed_dir, args.scene)
    save_dir = os.path.join(args.out_dir, args.scene)
    os.makedirs(save_dir, exist_ok=True)
    projections = np.load(os.path.join(scene_dir, 'projection', 'projections.npy'), allow_pickle=True).item()

    for file in tqdm(sorted(os.listdir(os.path.join(args.mask_dir, args.scene))), desc=args.scene):
        key = int(file.replace(".pt", ""))
        global_frame = key + int(projections[key].get("sub_frame", 0))
        raw_image = cv2.imread(os.path.join(scene_dir, 'video_all', f"{global_frame:04d}.jpg"))
        raw_image = cv2.resize(raw_image, (512, 512))
        mask = torch.load(os.path.join(args.mask_dir, args.scene, file))

        outfeat = pixel_features(raw_image, mask, visual_encoder, device)
        if outfeat is None:
            print(f"No valid masks found for {file}, skipping...")
            continue
        torch.save(outfeat, os.path.join(save_dir, file))


if __name__ == "__main__":
    main()
