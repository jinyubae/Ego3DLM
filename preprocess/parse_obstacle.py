"""
Per-frame obstacle parsing relative to the person's head pose.

Coordinate convention (verified from data):
  - World Z  = UP   (head Z std ~0.01m during walking)
  - Body col1 (body-Y) = FORWARD / gaze direction  (nearly horizontal, aligns with walk)
  - Body col0 (body-X) = RIGHT
  - All scene/head data reordered from storage YZX → world XYZ

Output per frame
────────────────
  FREE_FRONT  ∈ {LOW, MID, HIGH}       distance to nearest obstacle ahead
  FREE_LEFT   ∈ {LOW, MID, HIGH}
  FREE_RIGHT  ∈ {LOW, MID, HIGH}
  COLLIDE_STEP_FRONT ∈ {YES, NO}       would a 0.5 m step cause a body collision?
  COLLIDE_STEP_LEFT  ∈ {YES, NO}
  COLLIDE_STEP_RIGHT ∈ {YES, NO}
  BEST_DIR    ∈ {FRONT, LEFT, RIGHT, BACK}   direction with most free space

FREE thresholds (configurable):
  LOW  : free distance < LOW_TH  (1.0 m default)
  MID  : LOW_TH  ≤ dist < HIGH_TH (3.0 m default)
  HIGH : dist ≥ HIGH_TH

Usage:
  # Single sample
  python -m preprocess.parse_obstacle --sample_id 013579
  python -m preprocess.parse_obstacle --sample_id 013579 --verbose --save out.txt

  # All samples in the dataset (saved to --out_dir, default: obstacle_labels/)
  python -m preprocess.parse_obstacle --all
  python -m preprocess.parse_obstacle --all --out_dir /path/to/output --workers 8
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = "datasets/ego3dlm"
SCENE_DIR    = os.path.join(DATASET_ROOT, "3d_features_scene/voxelized_voxels")
HEAD_DIR     = os.path.join(DATASET_ROOT, "global_head_poses")
SCENE_INFO   = os.path.join(DATASET_ROOT, "scene_info_with_scene_index.txt")

# ── coordinate constants ───────────────────────────────────────────────────────
_YZX_TO_XYZ  = [2, 0, 1]   # storage YZX → world XYZ column reorder
WORLD_UP_IDX = 2            # world XYZ index for "up" (Z axis)
FWD_COL      = 1            # rotation column for forward  (body-Y)
RIGHT_COL    = 0            # rotation column for right    (body-X)

# ── thresholds ─────────────────────────────────────────────────────────────────
FREE_LOW_TH   = 1.0         # m  (below → LOW)
FREE_HIGH_TH  = 3.0         # m  (above → HIGH, between → MID)
CONE_DEG      = 60.0        # half-angle of directional sensing cone
HEIGHT_MARGIN = 1.0         # m above/below head to consider as obstacles
BODY_RADIUS   = 0.3         # m horizontal radius for collision cylinder


# ── I/O helpers ───────────────────────────────────────────────────────────────
def dequantize(voxel_indices, min_coords, voxel_size):
    pts_yzx = voxel_indices.astype(np.float32) * float(voxel_size) + min_coords.astype(np.float32)
    return pts_yzx[:, _YZX_TO_XYZ]


def load_scene(path):
    d = np.load(path, allow_pickle=True)
    pts = dequantize(d["voxel"], d["min_coords"], float(d["voxel_size"]))
    return pts, d["min_coords"], float(d["voxel_size"])


def load_head(path, min_coords, voxel_size):
    d           = np.load(path, allow_pickle=True)
    head_voxel  = dequantize(d["voxel"], min_coords, voxel_size)
    trans_exact = d["pose_yzx"][:, :3, 3].astype(np.float32)[:, _YZX_TO_XYZ]
    rotation    = d["rotation"][:, _YZX_TO_XYZ, :]
    return head_voxel, trans_exact, rotation


def build_scene_map():
    """Returns {sample_id: scene_name} and {scene_name: [sample_id, ...]}."""
    sample_to_scene = {}
    scene_to_samples = {}
    with open(SCENE_INFO) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            sid, sname = parts[0], parts[1]
            sample_to_scene[sid] = sname
            scene_to_samples.setdefault(sname, []).append(sid)
    return sample_to_scene, scene_to_samples


# ── horizontal projection ──────────────────────────────────────────────────────
def horiz(v):
    h = v.copy()
    h[WORLD_UP_IDX] = 0.0
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else h


# ── per-frame obstacle analysis ───────────────────────────────────────────────
def parse_frame(scene_pts, head_pos, head_rot, max_dist=5.0, step_dist=0.5):
    fwd_h   = horiz(head_rot[:, FWD_COL])
    right_h = horiz(head_rot[:, RIGHT_COL])
    left_h  = -right_h
    back_h  = -fwd_h

    dirs = {"FRONT": fwd_h, "RIGHT": right_h, "LEFT": left_h, "BACK": back_h}

    # height-filter
    dz          = scene_pts[:, WORLD_UP_IDX] - head_pos[WORLD_UP_IDX]
    near        = scene_pts[np.abs(dz) < HEIGHT_MARGIN]

    cos_thresh  = np.cos(np.radians(CONE_DEG))

    if len(near) == 0:
        free_dists = {d: max_dist for d in dirs}
    else:
        rel  = near - head_pos
        dist = np.linalg.norm(rel, axis=1)
        mask = (dist > 0.05) & (dist < max_dist)
        rel, dist = rel[mask], dist[mask]

        rel_h = rel.copy()
        rel_h[:, WORLD_UP_IDX] = 0.0
        rel_h_norm = rel_h / (np.linalg.norm(rel_h, axis=1, keepdims=True) + 1e-8)

        def free_distance(dir_vec):
            if len(dist) == 0:
                return max_dist
            in_cone = (rel_h_norm @ dir_vec) > cos_thresh
            return float(dist[in_cone].min()) if np.any(in_cone) else max_dist

        free_dists = {name: free_distance(d) for name, d in dirs.items()}

    def cat_free(d):
        if d < FREE_LOW_TH:  return "LOW"
        if d < FREE_HIGH_TH: return "MID"
        return "HIGH"

    def check_collide(dir_vec):
        new_pos  = head_pos + step_dist * dir_vec
        rel_s    = scene_pts - new_pos
        dz_s     = np.abs(rel_s[:, WORLD_UP_IDX])
        rel_s_h  = rel_s.copy(); rel_s_h[:, WORLD_UP_IDX] = 0.0
        dist_h   = np.linalg.norm(rel_s_h, axis=1)
        return "YES" if np.any((dz_s < HEIGHT_MARGIN) & (dist_h < BODY_RADIUS)) else "NO"

    collide   = {name: check_collide(d) for name, d in dirs.items()}
    best      = max(free_dists, key=free_dists.__getitem__)

    return {
        "FREE_FRONT":         cat_free(free_dists["FRONT"]),
        "FREE_LEFT":          cat_free(free_dists["LEFT"]),
        "FREE_RIGHT":         cat_free(free_dists["RIGHT"]),
        "COLLIDE_STEP_FRONT": collide["FRONT"],
        "COLLIDE_STEP_LEFT":  collide["LEFT"],
        "COLLIDE_STEP_RIGHT": collide["RIGHT"],
        "BEST_DIR":           best,
        "_dist_front": free_dists["FRONT"],
        "_dist_left":  free_dists["LEFT"],
        "_dist_right": free_dists["RIGHT"],
        "_dist_back":  free_dists["BACK"],
    }


def parse_sample(scene_pts, head_path, max_dist, step_dist, min_coords, voxel_size):
    """Parse all frames for one sample. Returns list of per-frame dicts."""
    _, trans, rotations = load_head(head_path, min_coords, voxel_size)
    return [parse_frame(scene_pts, trans[t], rotations[t], max_dist, step_dist)
            for t in range(len(trans))]


# ── formatting ─────────────────────────────────────────────────────────────────
def format_frame(t, res, verbose=False):
    lines = [
        f"[Frame {t:03d}]",
        f"  FREE_FRONT         = {res['FREE_FRONT']}",
        f"  FREE_LEFT          = {res['FREE_LEFT']}",
        f"  FREE_RIGHT         = {res['FREE_RIGHT']}",
        f"  COLLIDE_STEP_FRONT = {res['COLLIDE_STEP_FRONT']}",
        f"  COLLIDE_STEP_LEFT  = {res['COLLIDE_STEP_LEFT']}",
        f"  COLLIDE_STEP_RIGHT = {res['COLLIDE_STEP_RIGHT']}",
        f"  BEST_DIR           = {res['BEST_DIR']}",
    ]
    if verbose:
        lines.append(
            f"  (raw dist) front={res['_dist_front']:.2f}m  "
            f"left={res['_dist_left']:.2f}m  "
            f"right={res['_dist_right']:.2f}m  "
            f"back={res['_dist_back']:.2f}m"
        )
    return "\n".join(lines)


def format_summary(results, title=""):
    T = len(results)
    lines = ["=" * 60, f"SUMMARY  {title}", f"  {T} frames", "=" * 60]
    for key in ("FREE_FRONT", "FREE_LEFT", "FREE_RIGHT"):
        counts  = {}
        for r in results:
            counts[r[key]] = counts.get(r[key], 0) + 1
        dist_key = "_dist_" + key.split("_")[1].lower()
        dists    = [r[dist_key] for r in results]
        lines.append(
            f"  {key:20s}  LOW={counts.get('LOW',0):3d}  "
            f"MID={counts.get('MID',0):3d}  HIGH={counts.get('HIGH',0):3d}  "
            f"min={min(dists):.2f}m  mean={sum(dists)/len(dists):.2f}m"
        )
    lines.append("")
    for key in ("COLLIDE_STEP_FRONT", "COLLIDE_STEP_LEFT", "COLLIDE_STEP_RIGHT"):
        yes = sum(1 for r in results if r[key] == "YES")
        lines.append(f"  {key:30s}  YES={yes:3d}/{T}  NO={T-yes:3d}/{T}")
    lines.append("")
    best_counts = {}
    for r in results:
        best_counts[r["BEST_DIR"]] = best_counts.get(r["BEST_DIR"], 0) + 1
    lines.append("  BEST_DIR distribution:")
    for d in ("FRONT", "LEFT", "RIGHT", "BACK"):
        cnt = best_counts.get(d, 0)
        lines.append(f"    {d:6s}: {cnt:3d} frames  {'█' * cnt}")
    lines.append("=" * 60)
    return "\n".join(lines)


def make_header(title, n_scene_voxels, T, max_dist, step_dist):
    return (
        f"{'='*60}\n"
        f"OBSTACLE PARSE  {title}\n"
        f"  scene voxels : {n_scene_voxels:,}\n"
        f"  frames       : {T}\n"
        f"  max_dist     : {max_dist} m\n"
        f"  step_dist    : {step_dist} m\n"
        f"  cone         : ±{CONE_DEG}°\n"
        f"  height range : ±{HEIGHT_MARGIN} m around head\n"
        f"  body radius  : {BODY_RADIUS} m\n"
        f"  forward axis : body col {FWD_COL} (body-Y)\n"
        f"  right axis   : body col {RIGHT_COL} (body-X)\n"
        f"  world up     : world axis {WORLD_UP_IDX} (Z)\n"
        f"{'='*60}"
    )


def write_sample_output(out_path, title, scene_pts, results, max_dist, step_dist, verbose):
    header     = make_header(title, len(scene_pts), len(results), max_dist, step_dist)
    frame_text = "\n".join(format_frame(t, results[t], verbose) for t in range(len(results)))
    summary    = format_summary(results, title=title)
    with open(out_path, "w") as f:
        f.write("\n".join([header, "", frame_text, "", summary]) + "\n")


# ── worker for multiprocessing ────────────────────────────────────────────────
def _worker(args):
    """
    args = (sample_id, scene_name, out_dir, max_dist, step_dist)
    Loads scene & head data, parses all frames, writes output file.
    Returns (sample_id, ok, error_msg).
    """
    sample_id, scene_name, out_dir, max_dist, step_dist = args
    out_path = os.path.join(out_dir, f"{sample_id}.txt")
    if os.path.exists(out_path):
        return sample_id, True, "skipped (exists)"
    try:
        scene_path = os.path.join(SCENE_DIR, scene_name + ".npz")
        head_path  = os.path.join(HEAD_DIR,  sample_id + ".npz")
        scene_pts, min_coords, voxel_size = load_scene(scene_path)
        results    = parse_sample(scene_pts, head_path, max_dist, step_dist,
                                  min_coords, voxel_size)
        title      = f"{sample_id} | {scene_name}"
        write_sample_output(out_path, title, scene_pts, results,
                            max_dist, step_dist, verbose=False)
        return sample_id, True, "ok"
    except Exception as e:
        return sample_id, False, str(e)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # single-sample mode
    ap.add_argument("--sample_id", default=None)
    ap.add_argument("--scene",     default=None)
    ap.add_argument("--head",      default=None)
    ap.add_argument("--verbose",   action="store_true")
    ap.add_argument("--save",      default=None)
    # batch mode
    ap.add_argument("--all",       action="store_true",
                    help="Process every sample in the dataset")
    ap.add_argument("--out_dir",   default="obstacle_labels",
                    help="Output directory for batch mode (default: obstacle_labels/)")
    ap.add_argument("--workers",   type=int, default=4,
                    help="Parallel workers for batch mode (default: 4)")
    ap.add_argument("--resume",    action="store_true",
                    help="Skip already-processed samples (files that already exist)")
    # shared
    ap.add_argument("--max_dist",  type=float, default=5.0)
    ap.add_argument("--step",      type=float, default=0.5)
    args = ap.parse_args()

    # ── batch mode ────────────────────────────────────────────────────────────
    if args.all:
        sample_to_scene, _ = build_scene_map()

        # collect all sample IDs that have a head-pose file
        all_ids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(HEAD_DIR) if f.endswith(".npz")
        )
        # keep only those present in scene_info
        all_ids = [sid for sid in all_ids if sid in sample_to_scene]

        os.makedirs(args.out_dir, exist_ok=True)

        if args.resume:
            existing = {os.path.splitext(f)[0] for f in os.listdir(args.out_dir)}
            todo     = [sid for sid in all_ids if sid not in existing]
            print(f"Resuming: {len(todo)} remaining / {len(all_ids)} total")
        else:
            todo = all_ids

        print(f"Processing {len(todo)} samples → {args.out_dir}/  "
              f"(workers={args.workers})")

        work_args = [
            (sid, sample_to_scene[sid], args.out_dir, args.max_dist, args.step)
            for sid in todo
        ]

        n_ok, n_fail, n_skip = 0, 0, 0

        if args.workers == 1:
            # single-process with scene caching (most efficient for large batches)
            _, scene_to_samples = build_scene_map()

            # group todo by scene to minimise scene loads
            sid_set = set(todo)
            grouped = {}
            for sname, sids in scene_to_samples.items():
                batch = [s for s in sids if s in sid_set]
                if batch:
                    grouped[sname] = batch

            pbar = tqdm(total=len(todo), unit="sample")
            for sname, sids in grouped.items():
                # load scene once per group
                scene_path = os.path.join(SCENE_DIR, sname + ".npz")
                try:
                    scene_pts, min_coords, voxel_size = load_scene(scene_path)
                except Exception as e:
                    for sid in sids:
                        tqdm.write(f"  ERROR loading scene {sname}: {e}")
                    n_fail += len(sids)
                    pbar.update(len(sids))
                    continue

                for sid in sids:
                    out_path = os.path.join(args.out_dir, f"{sid}.txt")
                    if args.resume and os.path.exists(out_path):
                        n_skip += 1
                        pbar.update(1)
                        continue
                    try:
                        results = parse_sample(scene_pts,
                                               os.path.join(HEAD_DIR, sid + ".npz"),
                                               args.max_dist, args.step,
                                               min_coords, voxel_size)
                        write_sample_output(out_path, f"{sid} | {sname}",
                                            scene_pts, results,
                                            args.max_dist, args.step, verbose=False)
                        n_ok += 1
                    except Exception as e:
                        tqdm.write(f"  ERROR {sid}: {e}")
                        n_fail += 1
                    pbar.update(1)
            pbar.close()

        else:
            # multiprocessing (each worker loads its own scene)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, wa): wa[0] for wa in work_args}
                pbar = tqdm(as_completed(futures), total=len(futures), unit="sample")
                for fut in pbar:
                    sid, ok, msg = fut.result()
                    if ok and msg == "skipped (exists)":
                        n_skip += 1
                    elif ok:
                        n_ok += 1
                    else:
                        n_fail += 1
                        tqdm.write(f"  ERROR {sid}: {msg}")
                pbar.close()

        print(f"\nDone.  OK={n_ok}  skipped={n_skip}  failed={n_fail}")
        print(f"Output: {os.path.abspath(args.out_dir)}/")
        return

    # ── single-sample mode ────────────────────────────────────────────────────
    if args.sample_id is not None:
        sample_to_scene, _ = build_scene_map()
        scene_name = sample_to_scene[args.sample_id]
        scene_path = os.path.join(SCENE_DIR, scene_name + ".npz")
        head_path  = os.path.join(HEAD_DIR,  args.sample_id + ".npz")
        title      = f"Sample {args.sample_id}  |  {scene_name}"
    elif args.scene and args.head:
        scene_path, head_path = args.scene, args.head
        title = os.path.basename(args.head)
    else:
        ap.error("Provide --sample_id, --all, or both --scene and --head")

    scene_pts, min_coords, voxel_size = load_scene(scene_path)
    results = parse_sample(scene_pts, head_path, args.max_dist, args.step,
                           min_coords, voxel_size)

    header     = make_header(title, len(scene_pts), len(results), args.max_dist, args.step)
    frame_text = "\n".join(format_frame(t, results[t], args.verbose)
                           for t in range(len(results)))
    summary    = format_summary(results, title=title)
    full_out   = "\n".join([header, "", frame_text, "", summary])

    print(full_out)

    if args.save:
        with open(args.save, "w") as f:
            f.write(full_out + "\n")
        print(f"\nSaved → {args.save}")


if __name__ == "__main__":
    main()
