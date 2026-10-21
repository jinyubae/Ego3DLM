"""Extract per-segment egocentric video embeddings with the BLIP-2 EVA-ViT-g encoder.

For every segment folder ``<parsed_dir>/<scene>/video/NNNN/*.jpg`` (10 FPS) every 5th frame is encoded
(2 FPS) and the mean-pooled, L2-normalized patch embedding is stored:

    <out_dir>/<scene>/NNNN.pt   (num_frames, 1408)

Requires the LAVIS copy shipped with 3D-LLM (``three_steps_3d_feature/second_step``) with
``preprocess/scene/patches/eva_vit_dtype.patch`` applied, passed through ``--lavis_root``.

Usage:
    python -m preprocess.motion.video_features --parsed_dir /path/to/parsed_nymeria --out_dir /path/to/video_features_scene \
        --lavis_root /path/to/3D-LLM/three_steps_3d_feature/second_step
"""
import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512
FRAME_STRIDE = 5


@torch.no_grad()
def encode_batch(batch_tensors, visual_encoder, device):
    images = torch.stack(batch_tensors, dim=0).to(device=device, dtype=torch.float32)
    output = visual_encoder(images)
    if isinstance(output, (tuple, list)):
        output = output[0]
    feat = F.normalize(output.mean(1), dim=-1)
    return [f.cpu() for f in feat]


def process_sequence(sequence_dir, save_scene_dir, visual_encoder, device, batch_size, overwrite):
    save_file = save_scene_dir / f"{sequence_dir.name}.pt"
    if save_file.exists() and not overwrite:
        return

    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    frame_length = len(frame_paths) // FRAME_STRIDE * FRAME_STRIDE
    frame_paths = [frame_paths[i] for i in range(0, frame_length, FRAME_STRIDE)]

    features, batch = [], []
    for frame_path in frame_paths:
        raw_image = cv2.imread(str(frame_path))  # BGR, uint8, no normalization (as in 3D-LLM)
        if raw_image is None:
            print(f"Failed to read {frame_path}")
            continue
        raw_image = cv2.resize(raw_image, (LOAD_IMG_WIDTH, LOAD_IMG_HEIGHT))
        batch.append(torch.from_numpy(raw_image).permute(2, 0, 1).contiguous())
        if len(batch) >= batch_size:
            features.extend(encode_batch(batch, visual_encoder, device))
            batch = []
    if batch:
        features.extend(encode_batch(batch, visual_encoder, device))

    if features:
        torch.save(torch.stack(features, dim=0), save_file)


def main():
    parser = argparse.ArgumentParser(description="Extract per-segment EVA-ViT-g video features")
    parser.add_argument("--parsed_dir", required=True, help="root of the parsed Nymeria scenes")
    parser.add_argument("--out_dir", required=True, help="output root; features go to <out_dir>/<scene>/NNNN.pt")
    parser.add_argument("--lavis_root", required=True, help="folder that contains the (patched) `lavis` package")
    parser.add_argument("--scenes", nargs="+", default=None, help="subset of scenes (default: all)")
    parser.add_argument("--segments", nargs="+", default=None, help="subset of segment ids, e.g. 0001 0002")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.lavis_root)
    from lavis.models.eva_vit import create_eva_vit_g

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_encoder = create_eva_vit_g(512, precision="fp32").to(device=device, dtype=torch.float32)
    visual_encoder.eval()

    scene_root, save_root = Path(args.parsed_dir), Path(args.out_dir)
    scenes = args.scenes or sorted(p.name for p in scene_root.iterdir() if (p / "video").is_dir())
    for scene in tqdm(scenes, desc="scenes"):
        save_scene_dir = save_root / scene
        save_scene_dir.mkdir(parents=True, exist_ok=True)
        sequence_dirs = [p for p in sorted((scene_root / scene / "video").iterdir()) if p.is_dir()]
        if args.segments:
            sequence_dirs = [p for p in sequence_dirs if p.name in set(args.segments)]
        for sequence_dir in tqdm(sequence_dirs, desc=scene, leave=False):
            process_sequence(sequence_dir, save_scene_dir, visual_encoder, device, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
