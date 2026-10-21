import torch
import numpy as np


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def motion_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) == 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    return adapted_batch

def instruct_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1], dtype=torch.float32) for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
            "pc": collate_tensors([torch.as_tensor(b[9], dtype=torch.float32) for b in notnone_batches]),
            "pc_embed": collate_tensors([torch.as_tensor(b[10], dtype=torch.float32) for b in notnone_batches]),
            "pc_mask": collate_tensors([torch.as_tensor(b[11]).long() for b in notnone_batches]),
            "pc_length": [b[12] for b in notnone_batches],
            "three_points": collate_tensors([torch.as_tensor(b[13], dtype=torch.float32) for b in notnone_batches]),
            "three_points_length": [b[14] for b in notnone_batches],
            # "video_rgb": collate_tensors([torch.as_tensor(b[15]) for b in notnone_batches]),
            "video_embed": collate_tensors([torch.as_tensor(b[15], dtype=torch.float32) for b in notnone_batches]),
            "video_lengths": [b[16] for b in notnone_batches],
            "future_text": [b[17] for b in notnone_batches],
            "future_motion": collate_tensors([torch.as_tensor(b[18], dtype=torch.float32) for b in notnone_batches]),
            "future_motion_length": [b[19] for b in notnone_batches],
            "global_head_voxel": collate_tensors([torch.as_tensor(b[22], dtype=torch.float32) for b in notnone_batches]),
            "global_head_rot": collate_tensors([torch.as_tensor(b[23], dtype=torch.float32) for b in notnone_batches]),
            "min_coord": collate_tensors([torch.as_tensor(b[24], dtype=torch.float32) for b in notnone_batches]),
            "obstacle_cot_answer": [b[25] for b in notnone_batches],
            # "past_pose": collate_tensors([torch.as_tensor(b[20], dtype=torch.float32) for b in notnone_batches]),
            # "fut_pose": collate_tensors([torch.as_tensor(b[21], dtype=torch.float32) for b in notnone_batches]),
        })
        # if len(notnone_batches[0]) > 25 :
        #     adapted_batch.update({
        #     "scene_question": [b[25] for b in notnone_batches],
        #     "scene_answer": [b[26] for b in notnone_batches],
        #     })
    
    if not EvalFlag : 
        adapted_batch.update({
            "past_pose": collate_tensors([torch.as_tensor(b[20], dtype=torch.float32) for b in notnone_batches]),
            "fut_pose": collate_tensors([torch.as_tensor(b[21], dtype=torch.float32) for b in notnone_batches]),
        })
    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
            "scene_name": [b[26] for b in notnone_batches],
            "data_idx": [b[27] for b in notnone_batches],
        })

    # Tasks
    if notnone_batches[0][8] is not None:
    # if len(notnone_batches[0]) == 9+2:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    return adapted_batch


def scene_qa_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })
    
    # Text and scene
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "pc": collate_tensors([torch.as_tensor(b[9]) for b in notnone_batches]),
            "pc_embed": collate_tensors([torch.as_tensor(b[10]).float() for b in notnone_batches]),
            "pc_mask": collate_tensors([torch.as_tensor(b[11]).long() for b in notnone_batches]),
            "pc_len": [b[12] for b in notnone_batches],
            "video_embed": collate_tensors([torch.as_tensor(b[13]).float() for b in notnone_batches]),
            "video_lengths": [b[14] for b in notnone_batches],
            "scene_question": [b[15] for b in notnone_batches],
            "scene_answer": [b[16] for b in notnone_batches],
            "global_head_voxel": collate_tensors([torch.as_tensor(b[18], dtype=torch.float32) for b in notnone_batches]),
            "global_head_rot": collate_tensors([torch.as_tensor(b[19], dtype=torch.float32) for b in notnone_batches]),
            "min_coord": collate_tensors([torch.as_tensor(b[20], dtype=torch.float32) for b in notnone_batches]),
            "obstacle_answer": [b[21] if len(b) > 21 else '' for b in notnone_batches],
        })
    if not EvalFlag :
        adapted_batch.update({
            "past_pose": collate_tensors([torch.as_tensor(b[17], dtype=torch.float32) for b in notnone_batches]),
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if notnone_batches[0][8] is not None:
    # if len(notnone_batches[0]) == 9+2:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    return adapted_batch


