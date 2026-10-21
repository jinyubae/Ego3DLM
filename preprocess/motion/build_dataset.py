"""Assemble the flat Ego3DLM dataset (one file per 6-digit sample id) from the per-scene outputs.

Every sample id corresponds to one narrated 5 s segment of a scene; its future segment is the next id of the
same scene, so the last segment of every scene is not used as a sample. The released annotation bundle contains
the mapping ``scene_info_with_scene_index.txt`` (``id,scene,segment``); using it keeps the ids consistent with
the released splits, scene QA and obstacle labels.

    <out_dir>/new_joint_vecs/<id>.npy    <- <motion_dir>/new_joint_vecs/<scene>/<segment>.npy
    <out_dir>/new_joints/<id>.npy        <- <motion_dir>/new_joints/<scene>/<segment>.npy
    <out_dir>/three_points/<id>.npy      <- <motion_dir>/three_points/<scene>/<segment>.npy
    <out_dir>/video_features/<id>.pt     <- <video_dir>/<scene>/<segment>.pt
    <out_dir>/texts/<id>.txt             <- <parsed_dir>/<scene>/motion_caption/<segment>.txt
    <out_dir>/scene_info.txt             id,scene for the built ids

Run ``text_pos_tag.py`` afterwards to add the part-of-speech tags to ``texts/``.

Usage (reproduce the released dataset):
    python -m preprocess.motion.build_dataset --scene_index <bundle>/scene_info_with_scene_index.txt \
        --ids <bundle>/train_val.txt --motion_dir ... --video_dir ... --parsed_dir ... --out_dir datasets/ego3dlm

Usage (new data): pass ``--make_index --split_json data_split.json`` instead of ``--scene_index``.
"""
import argparse
import json
import shutil
from pathlib import Path

POS_PLACEHOLDER = "#data/NOUN#0.0#0.0"


def read_index(path):
    index = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sid, scene, segment = line.split(',')
                index.append((sid, scene, segment))
    return index


def make_index(motion_dir, split_json, start_id=0):
    """Number all segments (except the last one of each scene) of the scenes listed in the split, scene by scene."""
    with open(split_json) as f:
        split = json.load(f)
    scenes = set(split["train"]) | set(split["val"])
    index, gid = [], start_id
    for scene_dir in sorted((motion_dir / "new_joint_vecs").iterdir()):
        if scene_dir.name not in scenes:
            continue
        segments = sorted(p.stem for p in scene_dir.glob("*.npy"))
        for segment in segments[:-1]:
            index.append((f"{gid:06d}", scene_dir.name, segment))
            gid += 1
    return index, split


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", required=True, type=Path, help="output root of motion_representation.py")
    ap.add_argument("--video_dir", required=True, type=Path, help="output root of video_features.py")
    ap.add_argument("--parsed_dir", required=True, type=Path, help="root of the parsed Nymeria scenes (captions)")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--scene_index", type=Path, help="released scene_info_with_scene_index.txt")
    ap.add_argument("--ids", type=Path, help="optional file with the ids to build (one per line)")
    ap.add_argument("--make_index", action="store_true", help="create a new id mapping from --split_json")
    ap.add_argument("--split_json", type=Path, help="scene-level split {'train': [...], 'val': [...]}")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    split = None
    if args.make_index:
        index, split = make_index(args.motion_dir, args.split_json)
        with open(out / "scene_info_with_scene_index.txt", "w") as f:
            f.writelines(f"{sid},{scene},{segment}\n" for sid, scene, segment in index)
    else:
        index = read_index(args.scene_index)
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
        index = [e for e in index if e[0] in keep]

    missing_video = 0
    for sid, scene, segment in index:
        for name in ("new_joint_vecs", "new_joints", "three_points"):
            copy(args.motion_dir / name / scene / f"{segment}.npy", out / name / f"{sid}.npy")
        video_src = args.video_dir / scene / f"{segment}.pt"
        if video_src.exists():
            copy(video_src, out / "video_features" / f"{sid}.pt")
        else:
            missing_video += 1
        caption = (args.parsed_dir / scene / "motion_caption" / f"{segment}.txt").read_text().splitlines()
        (out / "texts").mkdir(parents=True, exist_ok=True)
        with open(out / "texts" / f"{sid}.txt", "w") as f:
            f.writelines(line.rstrip("\n") + POS_PLACEHOLDER + "\n" for line in caption)

    with open(out / "scene_info.txt", "w") as f:
        f.writelines(f"{sid},{scene}\n" for sid, scene, _ in index)

    if split is not None:
        ids = {"train": [], "val": []}
        for sid, scene, _ in index:
            ids["train" if scene in split["train"] else "val"].append(sid)
        for name, lst in (("train", ids["train"]), ("val", ids["val"]), ("test", ids["val"]),
                          ("train_val", sorted(ids["train"] + ids["val"]))):
            (out / f"{name}.txt").write_text("".join(f"{i}\n" for i in lst))

    print(f"built {len(index)} samples in {out}" + (f" ({missing_video} without video features)" if missing_video else ""))


if __name__ == "__main__":
    main()
