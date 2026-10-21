import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from transformers.utils import ModelOutput

from ego3dlm.config import instantiate_from_config, get_obj_from_str
from ego3dlm.losses.lm import LMLosses
from ego3dlm.metrics import LMMetrics
from ego3dlm.metrics.lm import M2TMetrics, MotionTextMetric, PredMetrics
from ego3dlm.archs.qformer import BertConfig, BertLMHeadModel
from .base import BaseModel

# The scene voxels are re-centred on the head of the last observed frame and shifted into a 64^3 grid
SHIFT_VOXEL = np.array([32, 32, 32], dtype=np.int32)
GRID_SIZE = 64

OBSTACLE_QUESTION = 'What obstacles are around the person in the current frame?'


@dataclass
class SimpleOutput(ModelOutput):
    loss: torch.Tensor = None


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1,
                keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def _normalize_voxel_size_np(voxel_size):
    size = np.asarray(voxel_size, dtype=np.float32)
    if size.ndim == 0:
        size = np.full(3, float(size), dtype=np.float32)
    return size


def _dequantize_np(voxel_indices, min_coords, voxel_size):
    indices = voxel_indices.astype(np.float32)
    min_coords = np.asarray(min_coords, dtype=np.float32).reshape(1, 3)
    size = np.asarray(voxel_size, dtype=np.float32).reshape(1, 3)
    return indices * size + min_coords


def _quantize_np(points, voxel_size):
    size = np.asarray(voxel_size, dtype=np.float32).reshape(1, 3)
    return np.round(points / size).astype(np.int32)


def _align_to_key_frame_np(scene_voxel, head_voxel, key_idx):
    key_voxel = head_voxel[key_idx]
    scene_aligned = scene_voxel - key_voxel[None, :]
    head_aligned = head_voxel - key_voxel[None, :]
    return key_voxel, scene_aligned, head_aligned


def _rotate_to_key_axes_np(scene_points, head_positions, head_rotations, key_idx):
    rotation_matrix = head_rotations[key_idx]
    rotated_scene = scene_points @ rotation_matrix.T
    rotated_positions = head_positions @ rotation_matrix.T
    rotated_rotations = head_rotations @ rotation_matrix.T
    return rotated_scene, rotated_positions, rotated_rotations, rotation_matrix


class Ego3DLM(BaseModel):
    """
    Ego3DLM: GPT-2 over text and PQ-VAE motion tokens, conditioned on the 3D scene (Q-Former tokens),
    the egocentric video and the three-point (head / wrists) tracking signal.
    Stage I   (lm_pretrain):  semantic and spatial scene QA
    Stage II  (lm_instruct):  scene reasoning -> current & future motion -> current & future narration
    Stage III (lm_grpo):      GRPO on the Stage II task
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='lm_pretrain',
                 debug=True,
                 task='motion_narration',
                 metrics_dict=['PredMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=True)
        self.datamodule = datamodule
        super().__init__()

        # Motion tokenizer (frozen) and motion-language model
        self.vae = instantiate_from_config(motion_vae)
        self.lm = instantiate_from_config(lm)
        self.lm.max_length = self.hparams.cfg.model.params.get('max_token_length', 256)

        # Q-Former for the 3D scene features; only its query branch is used
        self.Qformer, self.query_tokens = self.init_Qformer(self.hparams.cfg.model.params.num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        hidden_embedding_dim_lm = self.lm.language_model.config.hidden_size
        self.proj = nn.Linear(self.Qformer.config.hidden_size, hidden_embedding_dim_lm)

        # Three-point encoder: 6D rotation + translation of the head and both wrists; from Stage II on the
        # frame-to-frame relative poses are concatenated (model.params.rel_tp)
        tp_dim = (6 + 3) * 3
        if self.hparams.cfg.model.params.get("rel_tp", False):
            tp_dim *= 2
        self.three_points_proj = nn.Sequential(
            nn.Linear(tp_dim, hidden_embedding_dim_lm),
            nn.GELU(),
            nn.Linear(hidden_embedding_dim_lm, hidden_embedding_dim_lm),
            )

        # Positional encoding of the voxel coordinates, added to the scene features
        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 512, 1408 // 3)
        pos_encoding = pos_model(x).squeeze()
        self.register_buffer("pos_embedding", pos_encoding, persistent=False)

        # Projector of the video features
        self.img_proj = nn.Sequential(
            nn.Linear(1408, hidden_embedding_dim_lm),
            nn.GELU(),
            nn.Linear(hidden_embedding_dim_lm, hidden_embedding_dim_lm),
            )

        # GRPO reference model and its device (set by load_pretrained_grpo_ref in the train script)
        self.ref_model = None
        self.ref_device = None

        # The motion tokenizer stays frozen in all LM stages
        self.vae.training = False
        for p in self.vae.parameters():
            p.requires_grad = False

        self._losses = torch.nn.ModuleDict({
            split: LMLosses(cfg, self.datamodule.njoints)
            for split in ["losses_train", "losses_val"]
        })

        self.feats2joints = datamodule.feats2joints

        # Generations per validation sample (motion metrics use the best of them)
        self.num_sample_seq = 3

        # Normalization statistics of the three-point features (Stage II / III)
        self.tp_mean = self.tp_var = None
        if self.hparams.cfg.model.params.get("norm_tp", False):
            payload = torch.load(pjoin(self.hparams.cfg.DATASET.NYMERIA.ROOT, "tp_cat_stats.pt"), map_location="cpu")
            self.tp_mean = payload["mean"]
            self.tp_var = payload["var"]

    def configure_optimizers(self):
        cfg_optim = self.hparams.cfg.TRAIN.OPTIM
        optim_kwargs = dict(cfg_optim.params)
        base_lr = optim_kwargs.pop('lr', 1e-4)
        wd = optim_kwargs.pop('weight_decay', 0.0)

        # newly added modules (Q-Former and projectors) and the language model
        new_modules = ('Qformer.', 'query_tokens', 'proj.', 'three_points_proj.', 'img_proj.')
        qproj, other = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                (qproj if n.startswith(new_modules) else other).append(p)

        param_groups = []
        if qproj:
            param_groups.append({'params': qproj, 'lr': base_lr, 'weight_decay': wd, 'name': 'qformer_proj'})
        if other:
            param_groups.append({'params': other, 'lr': base_lr, 'weight_decay': wd, 'name': 'other_decay'})

        optim_target = cfg_optim.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(param_groups, **optim_kwargs)

        scheduler_target = self.hparams.cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.hparams.cfg.TRAIN.LR_SCHEDULER.params)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def configure_metrics(self):
        self.metrics = LMMetrics(datamodule=self.datamodule, **self.hparams)
        # Per-subtask metrics of the Stage II / III task, created on first use.
        # MotionTextMetrics must be placed before M2TMetrics so its compute()
        # runs first and can read the best_ADE_idx file before M2TMetrics removes it.
        self.subtask_metrics = {"PredMetrics": {}}
        if 'MotionTextMetrics' in self.hparams.cfg.METRIC.get('TYPE', []):
            self.subtask_metrics["MotionTextMetrics"] = {}
        self.subtask_metrics["M2TMetrics"] = {}

    def _get_or_create_subtask_metric(self, metric_name: str, subtask: str):
        """Per-subtask metric instance (e.g. PredMetrics of stp2mt_f), created on first use."""
        metric_map = self.subtask_metrics.setdefault(metric_name, {})
        if subtask not in metric_map:
            if metric_name == "PredMetrics":
                metric = PredMetrics(
                    cfg=self.hparams.cfg,
                    datamodule=self.datamodule,
                    diversity_times=30 if self.hparams.debug else self.hparams.cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=self.hparams.cfg.METRIC.DIST_SYNC_ON_STEP,
                    is_future='_f' in subtask,
                )
            elif metric_name == "M2TMetrics":
                metric = M2TMetrics(
                    cfg=self.hparams.cfg,
                    w_vectorizer=self.datamodule.hparams.w_vectorizer,
                    dist_sync_on_step=self.hparams.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric_name == "MotionTextMetrics":
                metric = MotionTextMetric(
                    cfg=self.hparams.cfg,
                    w_vectorizer=self.datamodule.hparams.w_vectorizer,
                    dist_sync_on_step=self.hparams.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
            metric_map[subtask] = metric.to(self.device)
        return metric_map[subtask]

    @staticmethod
    def rot_to_6d(R):  # R: (...,3,3)
        # Zhou et al. (CVPR'19) 6D rotation representation: concatenate the first two columns
        r1 = R[..., :, 0]  # (...,3)
        r2 = R[..., :, 1]  # (...,3)
        return torch.cat([r1, r2], dim=-1)  # (...,6)

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def add_positional_encoding_to_pc_embed(self, pc, pc_embed):
        device = pc_embed.device
        pos_embedding = self.pos_embedding
        if pos_embedding.device != device:
            pos_embedding = pos_embedding.to(device)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            all_pcs = torch.zeros_like(pc_embed)
            for j in range(pc.shape[0]):
                pcs = []
                for i in range(3):
                    try:
                        pc_i = pc[j][:, i].long()
                        pcs.append(pos_embedding.index_select(0, pc_i))
                    except Exception as e:
                        print(e)
                if pcs:
                    pcs = torch.cat(pcs, -1)
                    all_pcs[j, :, :pcs.shape[-1]] = pcs

            pc_embed = pc_embed + 0.01 * all_pcs
            return pc_embed

    def _prepare_scene_tokens(
        self,
        tasks,
        pc,             # (B,N,3) long voxel indices
        pc_embed,       # (B,N,D)
        pc_mask,        # (B,N) long/bool, 1=valid
        *,
        head_voxel,     # (B,T,3) long
        head_rotation,  # (B,T,3,3) float (world <- head)
        keyframe_index, # (B,) long/int
        min_coords,     # (B,3) float
        voxel_size,     # scalar float/tensor
    ):
        """Align the scene to the head pose of the key frame and encode it with the Q-Former.
        Returns one (num_query_token, H) tensor per sample (None for tasks without a scene)."""
        device = pc_embed.device

        if tasks is None:  # validation: every sample has a scene
            scene_flags = [True] * pc_embed.shape[0]
            num_tasks = pc_embed.shape[0]
        else:
            num_tasks = len(tasks)
            if pc is None or pc_embed is None or pc_mask is None:
                return [None] * num_tasks
            scene_flags = []
            for task in tasks:
                task_cls = ""
                if isinstance(task, dict):
                    task_cls = str(task.get("input", ""))
                scene_flags.append("<3d_scene_placeholder>" in task_cls.lower())

        if not any(scene_flags):
            return [None] * num_tasks

        scene_indices = torch.tensor(
            [idx for idx, flag in enumerate(scene_flags) if flag],
            device=device,
            dtype=torch.long,
        )

        pc_subset        = pc.index_select(0, scene_indices)              # (S,N,3)
        pc_embed_subset  = pc_embed.index_select(0, scene_indices)        # (S,N,D)
        pc_mask_subset   = pc_mask.index_select(0, scene_indices)         # (S,N)
        head_vox_subset  = head_voxel.index_select(0, scene_indices)      # (S,T,3)
        head_rot_subset  = head_rotation.index_select(0, scene_indices)   # (S,T,3,3)
        keyframe_index = torch.as_tensor(keyframe_index, dtype=torch.long, device=pc.device).view(-1)
        kf_idx_subset    = keyframe_index.index_select(0, scene_indices)  # (S,)
        min_subset       = min_coords.index_select(0, scene_indices)      # (S,3)

        S = pc_subset.shape[0]
        voxel_size_tensor = torch.as_tensor(voxel_size, dtype=torch.float32, device=device)
        voxel_size_np = _normalize_voxel_size_np(voxel_size_tensor.detach().cpu().numpy())

        pc_rot_list     = []
        pc_mask_list    = []
        pc_embed_list   = []

        for s in range(S):
            pc_s       = pc_subset[s]                  # (N,3) long
            pcemb_s    = pc_embed_subset[s]            # (N,D)
            pcmask_s   = pc_mask_subset[s].clone()     # (N,)
            heads_s    = head_vox_subset[s]            # (T,3)
            rots_s     = head_rot_subset[s]            # (T,3,3)
            kf_s       = int(kf_idx_subset[s].item())
            min_s      = min_subset[s]                 # (3,)

            pc_rot_s = pc_s.clone()
            valid_mask = pcmask_s.to(dtype=torch.bool)

            if not torch.any(valid_mask):
                pc_rot_list.append(pc_rot_s)
                pc_mask_list.append(pcmask_s)
                pc_embed_list.append(pcemb_s)
                continue

            scene_voxel_np = pc_s[valid_mask].detach().cpu().numpy()
            head_voxel_np = heads_s.detach().cpu().numpy()
            head_rot_np = rots_s.detach().cpu().numpy()
            min_np = min_s.detach().cpu().numpy()

            if scene_voxel_np.size == 0 or head_voxel_np.shape[0] == 0:
                pc_rot_list.append(pc_rot_s)
                pc_mask_list.append(pcmask_s)
                pc_embed_list.append(pcemb_s)
                continue

            key_idx = max(0, min(kf_s, head_voxel_np.shape[0] - 1))

            # translate to the key-frame head, rotate into its axes, re-voxelize and shift into the grid
            _, scene_voxel_aligned, head_voxel_aligned = _align_to_key_frame_np(
                scene_voxel_np, head_voxel_np, key_idx
            )
            zero_min = np.zeros_like(min_np, dtype=np.float32)
            scene_points_aligned = _dequantize_np(scene_voxel_aligned, zero_min, voxel_size_np)
            head_positions_aligned = _dequantize_np(head_voxel_aligned, zero_min, voxel_size_np)

            scene_points_rotated, _, _, _ = _rotate_to_key_axes_np(
                scene_points_aligned, head_positions_aligned, head_rot_np, key_idx
            )
            scene_voxel_rotated = _quantize_np(scene_points_rotated, voxel_size_np)
            scene_voxel_shifted = scene_voxel_rotated + SHIFT_VOXEL[None, :]

            pc_rot_s[valid_mask] = torch.from_numpy(scene_voxel_shifted).to(pc_rot_s.device, dtype=pc_rot_s.dtype)

            # drop the voxels that fall outside the grid
            invalid_mask = ((pc_rot_s < 0) | (pc_rot_s >= GRID_SIZE)).any(dim=-1)
            if int(invalid_mask.sum().item()) > 0:
                pc_rot_s[invalid_mask] = 0
                pcmask_s[invalid_mask] = 0

            pc_rot_list.append(pc_rot_s.to(torch.long))
            pc_mask_list.append(pcmask_s)
            pc_embed_list.append(pcemb_s)

        pc_rot_batch   = torch.stack(pc_rot_list, dim=0)       # (S,N,3)
        pc_mask_batch  = torch.stack(pc_mask_list, dim=0)      # (S,N)
        pc_embed_batch = torch.stack(pc_embed_list, dim=0)     # (S,N,D)

        pc_pe = self.add_positional_encoding_to_pc_embed(pc_rot_batch, pc_embed_batch)

        query_tokens = self.query_tokens.expand(pc_pe.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_pe,
            encoder_attention_mask=pc_mask_batch,
            return_dict=True,
        )
        projected_subset = self.proj(query_output.last_hidden_state)      # (S,Q,D')

        per_sample = [None] * num_tasks
        for local_idx, gidx in enumerate(scene_indices.tolist()):
            per_sample[gidx] = projected_subset[local_idx]
        return per_sample

    def _prepare_three_point_embeddings(self, three_points, num_samples, three_points_len):
        """Encode the head / wrist poses (B,T,3,4,4): absolute and frame-to-frame relative poses (B,T,54)."""
        if three_points is None:
            return [None] * num_samples
        assert three_points_len is not None

        B, T, P, _, _ = three_points.shape

        def _rel_identity_feat(B, P, device, dtype):
            rot6d_I = torch.tensor([1, 0, 0, 0, 1, 0], device=device, dtype=dtype).view(1, 1, 6)  # (1,1,6)
            t0 = torch.zeros((1, 1, 3), device=device, dtype=dtype)                                 # (1,1,3)
            per_point = torch.cat([rot6d_I, t0], dim=-1)                                            # (1,1,9)
            per_point = per_point.repeat(B, P, 1)                                                   # (B,P,9)
            return per_point.reshape(B, 1, -1)

        lens = torch.as_tensor(three_points_len, device=three_points.device, dtype=torch.long).clamp_(min=0, max=T)
        valid_abs = torch.arange(T, device=three_points.device).unsqueeze(0) < lens.unsqueeze(1)          # (B,T)
        valid_rel = torch.arange(T - 1, device=three_points.device).unsqueeze(0) < (lens - 1).unsqueeze(1) # (B,T-1)

        # ---------- ABS (B,T,27) ----------
        R_abs = three_points[..., :3, :3]
        t_abs = three_points[..., :3, 3]
        sixd_abs = self.rot_to_6d(R_abs.reshape(B, T, P, 3, 3)).reshape(B, T, P, 6)
        feat_abs = torch.cat([sixd_abs, t_abs], dim=-1)          # (B,T,P,9)
        x_abs = feat_abs.reshape(B, T, -1)                       # (B,T,27)

        # ---------- REL (B,T,27) ----------
        rel_1_to_T = three_points.new_zeros((B, T - 1, P, 4, 4))

        if valid_rel.any():
            prev = three_points[:, :-1]          # (B,T-1,P,4,4)
            curr = three_points[:,  1:]          # (B,T-1,P,4,4)

            prev_sel = prev[valid_rel]           # (N,P,4,4)
            curr_sel = curr[valid_rel]           # (N,P,4,4)

            # padding is excluded via valid_rel, so inv() never sees singular matrices.
            # torch.linalg.inv needs FP32: compute the inverse in FP32, then cast back to the destination dtype.
            prev_inv_sel = torch.linalg.inv(prev_sel.float())
            rel_sel = (prev_inv_sel @ curr_sel.float()).to(rel_1_to_T.dtype)

            rel_1_to_T[valid_rel] = rel_sel

        R_rel = rel_1_to_T[..., :3, :3]                          # (B,T-1,P,3,3)
        t_rel = rel_1_to_T[..., :3, 3]                           # (B,T-1,P,3)
        sixd_rel = self.rot_to_6d(R_rel.reshape(B, T - 1, P, 3, 3)).reshape(B, T - 1, P, 6)
        feat_rel = torch.cat([sixd_rel, t_rel], dim=-1)          # (B,T-1,P,9)
        x_rel = feat_rel.reshape(B, T - 1, -1)                   # (B,T-1,27)

        # prepend identity-delta at t=0 -> (B,T,27)
        id0 = _rel_identity_feat(B, P, x_rel.device, x_rel.dtype)  # (B,1,27)
        x_rel_pad = torch.cat([id0, x_rel], dim=1)                 # (B,T,27)

        # ---------- CONCAT (B,T,54) ----------
        x = torch.cat([x_abs, x_rel_pad], dim=-1)                # (B,T,54)

        # normalize with the dataset statistics; padding stays zero
        if self.tp_mean is not None:
            mean = self.tp_mean.to(x.device, dtype=x.dtype)
            std = torch.sqrt(self.tp_var.to(x.device, dtype=x.dtype) + 1e-6)
            x = (x - mean) / std
            x = x.masked_fill(~valid_abs.unsqueeze(-1), 0.0)

        return self.three_points_proj(x)

    def _build_future_motion_strings(self, future_motion, future_motion_lengths):
        if future_motion is None or future_motion_lengths is None:
            return None
        return self.lm.motion_token_to_string(future_motion.long(), future_motion_lengths)

    def _prepare_video_tokens(self, video_embed, target_device=None):
        if video_embed is None:
            return None
        if target_device is None:
            target_device = self.query_tokens.device
        return self.img_proj(video_embed.to(target_device))

    def _scene_inputs(self, batch, tasks, keyframe_index):
        """Q-Former scene tokens of a batch, aligned to the head pose at `keyframe_index`."""
        return self._prepare_scene_tokens(tasks, batch["pc"], batch["pc_embed"], batch["pc_mask"],
                                          head_voxel=batch["global_head_voxel"],
                                          head_rotation=batch["global_head_rot"],
                                          keyframe_index=keyframe_index,
                                          min_coords=batch["min_coord"],
                                          voxel_size=0.15)

    # ------------------------------------------------------------------ GRPO reward helpers
    def _repeat_modality(self, modality, repeat):
        if modality is None:
            return None
        if isinstance(modality, torch.Tensor):
            return modality.repeat_interleave(repeat, dim=0)
        return [item for item in modality for _ in range(repeat)]

    def _extract_motion_tokens_from_text(self, text: str):
        if "<motion_start>" not in text or "<motion_end>" not in text:
            return None, text
        middle = text.split("<motion_start>", 1)[1].split("<motion_end>", 1)[0]
        matches = re.findall(r"<motion_id_(\d+)>", middle)
        if not matches:
            return None, text
        num_codebooks = int(getattr(self.lm, "m_codebook_num", 1))
        valid_tokens = (len(matches) // num_codebooks) * num_codebooks
        if valid_tokens == 0:
            return None, text
        matches = matches[:valid_tokens]
        T = valid_tokens // num_codebooks
        tokens = torch.zeros((T, num_codebooks), dtype=torch.long, device=self.device)
        for t in range(T):
            for cb in range(num_codebooks):
                tokens[t, cb] = int(matches[t * num_codebooks + cb])
        cleaned = text.replace(middle, "<Motion_Placeholder>")
        return tokens, cleaned

    def _clean_text_for_bleu(self, text: str):
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _bleu_score(self, pred: str, ref: str, max_n: int = 4):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not pred_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
            ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
            overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in pred_ngrams.items())
            total = max(sum(pred_ngrams.values()), 1)
            precisions.append((overlap + 1.0) / (total + 1.0))

        ref_len = len(ref_tokens)
        pred_len = len(pred_tokens)
        if pred_len < ref_len:
            bp = math.exp(1.0 - (ref_len / max(pred_len, 1)))
        else:
            bp = 1.0
        return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)

    @torch.no_grad()
    def _compute_matching_dist(self, gt_text, pred_text, gt_motion_token, pred_motion_token):
        """Text-motion matching distances of the evaluator: (gt text, pred motion), (pred text, gt motion),
        (pred text, pred motion)."""
        if pred_motion_token is None and pred_text is None:
            return None, None, None

        get_text_embedding = getattr(self.metrics, 'M2TMetrics')._get_text_embeddings
        get_motion_embeddings = getattr(self.metrics, 'PredMetrics').get_motion_embeddings

        gt_motion = self.vae.decode(gt_motion_token.int())
        gt_motion_emb = get_motion_embeddings(gt_motion, [gt_motion.shape[1]])
        gt_text_emb = get_text_embedding([gt_text])

        gt_t_pred_m, pred_t_gt_m, pred_t_pred_m = 0, 0, 0
        if pred_motion_token is not None:
            pred_motion = self.vae.decode(pred_motion_token.int())
            pred_motion_emb = get_motion_embeddings(pred_motion, [pred_motion.shape[1]])
            gt_t_pred_m = euclidean_distance_matrix(gt_text_emb, pred_motion_emb).nan_to_num()

        if pred_text is not None:
            pred_text_emb = get_text_embedding([pred_text])
            pred_t_gt_m = euclidean_distance_matrix(pred_text_emb, gt_motion_emb).nan_to_num()

        if pred_motion_token is not None and pred_text is not None:
            pred_t_pred_m = euclidean_distance_matrix(pred_text_emb, pred_motion_emb).nan_to_num()

        return gt_t_pred_m, pred_t_gt_m, pred_t_pred_m

    @torch.no_grad()
    def _compute_ade(self, pred_tokens, gt_tokens):
        """Head ADE between the decoded motions (meters)."""
        if pred_tokens is None or gt_tokens is None:
            return None
        pred_tokens = pred_tokens.to(self.device)
        gt_tokens = gt_tokens.to(self.device).type(torch.int64)
        pred_joints = self.feats2joints(self.vae.decode(pred_tokens))
        gt_joints = self.feats2joints(self.vae.decode(gt_tokens))
        min_len = min(pred_joints.shape[1], gt_joints.shape[1])
        if min_len == 0:
            return None
        diff = (pred_joints[:, :min_len] - gt_joints[:, :min_len])[:, :, 6]
        return torch.linalg.norm(diff, dim=-1).mean()

    def on_save_checkpoint(self, checkpoint):
        # The frozen GRPO reference model is reloaded from disk via
        # load_pretrained_grpo_ref on every run, so persisting its weights inside
        # the training checkpoint doubles file size for no benefit.
        sd = checkpoint.get("state_dict")
        if sd is not None:
            checkpoint["state_dict"] = {k: v for k, v in sd.items() if not k.startswith("ref_model.")}

    # ------------------------------------------------------------------ training
    def train_lm_forward_pretraining(self, batch, batch_idx):
        """Stage I: semantic and spatial (obstacle) scene QA."""
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]

        input_motion_lengths = [int(length*self.lm.predict_ratio)*self.lm.down_t for length in lengths]
        keyframe_index = [int(l - 1) for l in input_motion_lengths]

        inputs_3d_scene = self._scene_inputs(batch, tasks, keyframe_index)
        inputs_video = self._prepare_video_tokens(batch["video_embed"], target_device=tokens_ref.device)

        # obstacleqa samples are answered with the obstacle label
        obstacle_answers = batch.get('obstacle_answer', [None] * len(texts))
        scene_a = []
        for idx, task in enumerate(tasks):
            task_class = task.get('class', '') if isinstance(task, dict) else ''
            if task_class == 'obstacleqa' and obstacle_answers[idx]:
                scene_a.append(obstacle_answers[idx])
            else:
                scene_a.append(batch['scene_answer'][idx])

        outputs = self.lm(
            texts=texts,
            motion_tokens=tokens_ref,
            lengths=lengths,
            tasks=tasks,
            pc_token_embeds=inputs_3d_scene,
            video_token_embeds=inputs_video,
            video_lengths=batch["video_lengths"],
            scene_q=batch['scene_question'],
            scene_a=scene_a,
        )
        return {'outputs': outputs}

    def train_lm_forward_instruction_tuning(self, batch, batch_idx):
        """Stage II: scene reasoning -> current & future motion -> current & future narration."""
        tokens_ref = batch["motion"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        three_points_len = batch["three_points_length"]

        input_motion_lengths = [int(length*self.lm.down_t) for length in lengths]
        keyframe_index = [int(l*self.lm.predict_ratio) - 1 for l in input_motion_lengths]

        pc_token_embeds = self._scene_inputs(batch, tasks, keyframe_index)
        inputs_video = self._prepare_video_tokens(batch["video_embed"], target_device=tokens_ref.device)
        future_motion_strings = self._build_future_motion_strings(batch["future_motion"], batch["future_motion_length"])
        three_points_embed = self._prepare_three_point_embeddings(batch["three_points"], len(tasks), three_points_len)

        outputs = self.lm(
            texts=batch["text"],
            motion_tokens=tokens_ref,
            lengths=lengths,
            tasks=tasks,
            pc_token_embeds=pc_token_embeds,
            video_token_embeds=inputs_video,
            video_lengths=batch["video_lengths"],
            future_motion_strings=future_motion_strings,
            future_texts=batch["future_text"],
            three_points_embeds=three_points_embed,
            three_points_lengths=three_points_len,
            obstacle_cot_answer=batch['obstacle_cot_answer'],
        )
        return {'outputs': outputs}

    def train_lm_forward_grpo(self, batch, batch_idx):
        """Stage III: GRPO with motion (ADE), text (BLEU-4) and text-motion matching rewards."""
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        video_lengths = batch["video_lengths"]
        three_points_len = batch["three_points_length"]
        future_motion = batch["future_motion"]
        future_motion_lengths = batch["future_motion_length"]
        future_texts = batch["future_text"]

        input_motion_lengths = [int(length*self.lm.down_t) for length in lengths]
        keyframe_index = [int(l*self.lm.predict_ratio) - 1 for l in input_motion_lengths]

        pc_token_embeds = self._scene_inputs(batch, tasks, keyframe_index)
        inputs_video = self._prepare_video_tokens(batch["video_embed"], target_device=tokens_ref.device)
        future_motion_strings = self._build_future_motion_strings(future_motion, future_motion_lengths)
        three_points_embed = self._prepare_three_point_embeddings(batch["three_points"], len(tasks), three_points_len)

        motion_strings = self.lm.motion_token_to_string(tokens_ref, lengths)
        prompts, _ = self.lm.template_fulfill(
            tasks,
            motion_strings,
            texts,
            future_motion_strings=future_motion_strings,
            future_texts=future_texts
        )

        grpo = self.hparams.cfg.TRAIN.GRPO
        num_samples = int(grpo.get("NUM_SAMPLES", 6))
        max_new_tokens = int(grpo.get("MAX_NEW_TOKENS", 128))
        temperature = float(grpo.get("TEMPERATURE", 1.0))
        top_p = float(grpo.get("TOP_P", 0.9))
        top_k = int(grpo.get("TOP_K", 50))
        beta = float(grpo.get("BETA", 0.001))
        eps = float(grpo.get("EPSILON", 0.2))
        motion_weight = float(grpo.get("MOTION_WEIGHT", 1.0))
        text_weight = float(grpo.get("TEXT_WEIGHT", 1.0))
        matching_dist_weight = float(grpo.get("MATCHING_DIST_WEIGHT", 1.0))
        missing_motion_penalty = float(grpo.get("MISSING_MOTION_PENALTY", 100.0))

        # Sample a group of outputs per prompt (the policy stays in train mode)
        with torch.no_grad():
            outputs_texts = self.lm.sample_group(
                prompts=prompts,
                pc_token_embeds=pc_token_embeds,
                video_token_embeds=inputs_video,
                video_lengths=video_lengths,
                three_points_embed=three_points_embed,
                three_points_lengths=three_points_len,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        if not outputs_texts:
            dummy_loss = torch.tensor(0.0, device=tokens_ref.device, requires_grad=True)
            return {'outputs': SimpleOutput(loss=dummy_loss)}

        expanded_prompts = [prompt for prompt in prompts for _ in range(num_samples)]
        pc_rep = self._repeat_modality(pc_token_embeds, num_samples)
        video_rep = self._repeat_modality(inputs_video, num_samples)
        video_len_rep = self._repeat_modality(video_lengths, num_samples)
        tp_rep = self._repeat_modality(three_points_embed, num_samples)
        tp_len_rep = self._repeat_modality(three_points_len, num_samples)

        # Current policy logprobs (eval mode for deterministic, dropout-free logprobs)
        self.lm.eval()
        logprobs = self.lm.compute_logprobs(
            prompts=expanded_prompts,
            outputs_text=outputs_texts,
            pc_token_embeds=pc_rep,
            video_token_embeds=video_rep,
            video_lengths=video_len_rep,
            three_points_embed=tp_rep,
            three_points_lengths=tp_len_rep,
        )
        self.lm.train()

        # Reference model logprobs
        if self.ref_model is None:
            raise RuntimeError("GRPO needs a reference model (TRAIN.GRPO.REF_MODEL_PATH)")
        with torch.no_grad():
            self.ref_model.lm.eval()
            logprobs_ref = self.ref_model.lm.compute_logprobs(
                prompts=expanded_prompts,
                outputs_text=outputs_texts,
                pc_token_embeds=pc_rep,
                video_token_embeds=video_rep.to(self.ref_device),
                video_lengths=video_len_rep,
                three_points_embed=tp_rep.to(self.ref_device),
                three_points_lengths=tp_len_rep,
            )
        batch_size = len(prompts)
        logprobs_ref = logprobs_ref.to(logprobs.device)

        log_diff = torch.clamp(logprobs - logprobs_ref.detach(), min=-2.0, max=2.0)
        ratio = torch.exp(log_diff)
        ratio = ratio.view(batch_size, num_samples)

        kl_penalty_raw = torch.mean(
            torch.exp(logprobs_ref.detach() - logprobs) -
            (logprobs_ref.detach() - logprobs) - 1
        )
        kl_penalty = beta * kl_penalty_raw

        # Rewards
        rewards = torch.zeros((batch_size, num_samples), device=tokens_ref.device)
        log_vals = {k: [] for k in ["raw_ade", "raw_bleu_past", "raw_bleu_future", "raw_matdist_past",
                                    "raw_matdist_future", "reward_motion", "reward_past_text",
                                    "reward_future_text", "reward_past_matdist", "reward_future_matdist"]}

        for i in range(batch_size):
            gt_text_c = texts[i]
            gt_text_f = future_texts[i] if future_texts is not None else ''
            gt_motion_tokens_c = tokens_ref[i][:lengths[i]] if tokens_ref is not None else None
            gt_motion_tokens_f = future_motion[i][:future_motion_lengths[i]] if future_motion is not None else None
            gt_motion_tokens = torch.cat((gt_motion_tokens_c, gt_motion_tokens_f), dim=0)
            cur_tok_length = lengths[i]

            for s in range(num_samples):
                pred_motion_tokens, pred_text_clean = self._extract_motion_tokens_from_text(
                    outputs_texts[i * num_samples + s])

                # Split the predicted motion into current and future
                if pred_motion_tokens is not None and pred_motion_tokens.shape[0] > cur_tok_length:
                    pred_motion_tokens_c = pred_motion_tokens[:cur_tok_length]
                    pred_motion_tokens_f = pred_motion_tokens[cur_tok_length:]
                elif pred_motion_tokens is not None:
                    pred_motion_tokens_c = pred_motion_tokens
                    pred_motion_tokens_f = None
                else:
                    pred_motion_tokens_c = None
                    pred_motion_tokens_f = None

                # Current and future descriptions
                cleaned = pred_text_clean.replace('<|endoftext|>', '')
                cur_match = re.search(r"<CUR_MOTION_DESC>(.*?)<CUR_MOTION_DESC/>", cleaned, flags=re.DOTALL)
                fut_match = re.search(r"<FUT_MOTION_DESC>(.*?)<FUT_MOTION_DESC/>", cleaned, flags=re.DOTALL)
                pred_text_c = cur_match.group(1).strip() if cur_match else ''
                pred_text_f = fut_match.group(1).strip() if fut_match else ''

                format_penalty = 0.0
                if cleaned.count("<CUR_MOTION_DESC>") != 1 or cleaned.count("<CUR_MOTION_DESC/>") != 1:
                    format_penalty -= 1.0
                if cleaned.count("<FUT_MOTION_DESC>") != 1 or cleaned.count("<FUT_MOTION_DESC/>") != 1:
                    format_penalty -= 1.0
                if cleaned.count("<motion_start>") != 1 or cleaned.count("<motion_end>") != 1:
                    format_penalty -= 1.0

                bleu_c = 0.0
                if gt_text_c:
                    bleu_c = self._bleu_score(self._clean_text_for_bleu(pred_text_c), self._clean_text_for_bleu(gt_text_c))
                bleu_f = 0.0
                if gt_text_f:
                    bleu_f = self._bleu_score(self._clean_text_for_bleu(pred_text_f), self._clean_text_for_bleu(gt_text_f))

                # Head ADE over the current and future motion together
                ade = self._compute_ade(pred_motion_tokens, gt_motion_tokens)
                if ade is None:
                    ade = missing_motion_penalty

                gt_t_pred_m_c, pred_t_gt_m_c, pred_t_pred_m_c = self._compute_matching_dist(gt_text_c, pred_text_c, gt_motion_tokens_c, pred_motion_tokens_c)
                if gt_t_pred_m_c + pred_t_gt_m_c + pred_t_pred_m_c == 0:
                    matching_dist_c = -missing_motion_penalty
                else:
                    matching_dist_c = -(gt_t_pred_m_c + pred_t_gt_m_c + pred_t_pred_m_c)

                gt_t_pred_m_f, pred_t_gt_m_f, pred_t_pred_m_f = self._compute_matching_dist(gt_text_f, pred_text_f, gt_motion_tokens_f, pred_motion_tokens_f)
                if gt_t_pred_m_f + pred_t_gt_m_f + pred_t_pred_m_f == 0:
                    matching_dist_f = -missing_motion_penalty
                else:
                    matching_dist_f = -(gt_t_pred_m_f + pred_t_gt_m_f + pred_t_pred_m_f)

                r_motion = motion_weight * max(0, 1 - float(ade))
                r_past_text = text_weight * bleu_c
                r_future_text = text_weight * bleu_f
                r_past_matdist = matching_dist_weight * matching_dist_c
                r_future_matdist = matching_dist_weight * matching_dist_f
                rewards[i, s] = r_motion + r_past_text + r_future_text + r_past_matdist + r_future_matdist + format_penalty

                for k, v in (("raw_ade", ade), ("raw_bleu_past", bleu_c), ("raw_bleu_future", bleu_f),
                             ("raw_matdist_past", matching_dist_c), ("raw_matdist_future", matching_dist_f),
                             ("reward_motion", r_motion), ("reward_past_text", r_past_text),
                             ("reward_future_text", r_future_text), ("reward_past_matdist", r_past_matdist),
                             ("reward_future_matdist", r_future_matdist)):
                    log_vals[k].append(float(v))

        # Clipped policy objective with group-normalized advantages
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        advantages = advantages / (rewards.std(dim=1, keepdim=True) + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
        unclipped_objective = ratio * advantages
        clipped_objective = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.minimum(unclipped_objective, clipped_objective))
        policy_loss = torch.clamp(policy_loss, min=-10.0, max=10.0)
        loss = policy_loss + kl_penalty

        if self.trainer.is_global_zero:
            grpo_log = {
                "grpo/reward_total": rewards.mean().item(),
                "grpo/kl_penalty_raw": kl_penalty_raw.item(),
                "grpo/policy_loss": policy_loss.item(),
                "grpo/total_loss": loss.item(),
            }
            grpo_log.update({f"grpo/{k}": sum(v) / len(v) for k, v in log_vals.items() if v})
            # Use Lightning logger step to keep WandB steps monotonic.
            self.log_dict(
                grpo_log,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )

        return {'outputs': SimpleOutput(loss=loss)}

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def val_sceneqa_obstacleqa_forward(self, batch, batch_idx):
        """Stage I validation: answer the semantic scene question and the obstacle question."""
        lengths = batch["length"]
        scene_answers = batch['scene_answer']

        token_lengths_ = [length // self.lm.down_t for length in lengths]
        input_motion_lengths_ = [int(token_length_ * self.lm.predict_ratio) * self.lm.down_t for token_length_ in token_lengths_]
        keyframe_index = [int(l - 1) for l in input_motion_lengths_]

        inputs_3d_scene = self._scene_inputs(batch, None, keyframe_index)
        inputs_video = self._prepare_video_tokens(batch["video_embed"], target_device=batch["pc_embed"].device)

        cleaned_text_sceneqa = self.lm.answer_scene_questions(batch['scene_question'], inputs_3d_scene,
                                                              inputs_video, batch["video_lengths"])
        cleaned_text_obstacleqa = self.lm.answer_scene_questions([OBSTACLE_QUESTION] * len(scene_answers),
                                                                 inputs_3d_scene, inputs_video, batch["video_lengths"])
        return {
            "t_ref_scene": scene_answers,
            "t_rst_scene": cleaned_text_sceneqa,
            "t_ref_obstacle": batch['obstacle_answer'],
            "t_rst_obstacle": cleaned_text_obstacleqa,
        }

    @torch.no_grad()
    def val_motion_and_text_forward(self, batch, batch_idx):
        """Stage II / III validation and evaluation: motion tracking (current, '_c') and prediction
        (future, '_f') with their descriptions, `num_sample_seq` generations per sample."""
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        future_motion = batch["future_motion"]
        future_motion_lengths = batch["future_motion_length"]
        future_texts = batch["future_text"]
        three_points_len = batch["three_points_length"]
        n_seq = self.num_sample_seq

        token_lengths_ = [length // self.lm.down_t for length in lengths]
        input_motion_lengths_ = [int(token_length_ * self.lm.predict_ratio) * self.lm.down_t for token_length_ in token_lengths_]
        keyframe_index = [int(l - 1) for l in input_motion_lengths_]

        inputs_3d_scene = self._scene_inputs(batch, None, keyframe_index)
        inputs_video = self._prepare_video_tokens(batch["video_embed"], target_device=feats_ref.device)
        three_points_embed = self._prepare_three_point_embeddings(batch["three_points"], len(feats_ref), three_points_len)

        ref_texts_f_list = []
        for entry in future_texts:
            if entry is None:
                ref_texts_f_list.append('')
            elif isinstance(entry, (list, tuple)):
                ref_texts_f_list.append(' '.join(str(x) for x in entry))
            else:
                ref_texts_f_list.append(str(entry))
        ref_texts_c_list = texts

        outputs_motions, generated_texts = self.lm.generate_motion_and_text(
            len(lengths),
            inputs_3d_scene,
            inputs_video,
            batch["video_lengths"],
            three_points_embed,
            three_points_len,
            n_seq,
        )

        ref_lengths_c_list = [int(l) for l in lengths]
        ref_lengths_f_list = [int(l) for l in future_motion_lengths]

        # Split the generated motion into current and future
        outputs_motion_c, outputs_motion_f = [], []
        for i, motions in enumerate(outputs_motions):
            cur_tok_length = ref_lengths_c_list[i // n_seq] // self.lm.down_t
            if motions.shape[0] > cur_tok_length:
                outputs_motion_c.append(motions[:cur_tok_length])
                outputs_motion_f.append(motions[cur_tok_length:])
            else:
                outputs_motion_c.append(motions)
                outputs_motion_f.append(torch.zeros((1, 2), device=motions.device))

        outputs_text_c, outputs_text_f = [], []
        for text in generated_texts:
            text = text.replace('<|endoftext|>', '')
            cur = re.search(r"<CUR_MOTION_DESC>(.*?)<CUR_MOTION_DESC/>", text)
            fut = re.search(r"<FUT_MOTION_DESC>(.*?)<FUT_MOTION_DESC/>", text)
            outputs_text_c.append(cur.group(1) if cur else '')
            outputs_text_f.append(fut.group(1) if fut else '')

        # Motion decode
        feats_rst_c = torch.zeros_like(feats_ref.repeat(n_seq, 1, 1))
        feats_rst_f = torch.zeros_like(future_motion.repeat(n_seq, 1, 1))
        min_len_c = []
        min_len_f = []
        for i in range(feats_rst_c.shape[0]):
            ref_index = i // n_seq
            outputs_motion_c[i] = torch.clamp(outputs_motion_c[i], 0, self.hparams.codebook_size - 1)
            outputs_motion_f[i] = torch.clamp(outputs_motion_f[i], 0, self.hparams.codebook_size - 1)

            if len(outputs_motion_c[i]) > 1:
                motion_c = self.vae.decode(outputs_motion_c[i])
            else:
                motion_c = torch.zeros_like(feats_ref[ref_index:ref_index + 1, ...])
            if len(outputs_motion_f[i]) > 1:
                motion_f = self.vae.decode(outputs_motion_f[i])
            else:
                motion_f = torch.zeros_like(future_motion[ref_index:ref_index + 1, ...])

            cur_len_c = min(motion_c.shape[1], ref_lengths_c_list[ref_index])
            cur_len_f = min(motion_f.shape[1], ref_lengths_f_list[ref_index])
            feats_rst_c[i:i + 1, :cur_len_c] = motion_c[:, :cur_len_c]
            feats_rst_f[i:i + 1, :cur_len_f] = motion_f[:, :cur_len_f]
            min_len_c.append(cur_len_c)
            min_len_f.append(cur_len_f)

        # Joints for the motion metrics, renormalized features for the text-motion evaluator
        subtasks = {
            'stp2mt_c': (feats_ref, feats_rst_c, min_len_c, ref_texts_c_list, outputs_text_c),
            'stp2mt_f': (future_motion, feats_rst_f, min_len_f, ref_texts_f_list, outputs_text_f),
        }
        rs_set = {"m2m": {}, "m2t": {}, "motion_text": {}}
        for subtask, (ref, rst, min_len, ref_texts, pred_texts) in subtasks.items():
            rs_set["m2m"][subtask] = {
                "m_ref": self.datamodule.renorm_for_evaluator(ref),
                "m_rst": self.datamodule.renorm_for_evaluator(rst),
                "joints_ref": self.feats2joints(ref),
                "joints_rst": self.feats2joints(rst),
                "length": min_len,
            }
        for subtask, (ref, rst, min_len, ref_texts, pred_texts) in subtasks.items():
            rs_set["m2t"][subtask] = {
                "t_ref": ref_texts,
                "t_pred": pred_texts,
                "feats_ref": rs_set["m2m"][subtask]["m_ref"],
                "length": min_len,
            }
        # Motion-text self-consistency: predicted motion vs predicted text
        for subtask, (ref, rst, min_len, ref_texts, pred_texts) in subtasks.items():
            rs_set["motion_text"][subtask] = {
                "feats_rst": rs_set["m2m"][subtask]["m_rst"],
                "t_pred": pred_texts,
                "length": min_len,
            }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # PredMetrics writes the best-of-n sample indices here; M2T / MotionText metrics read them
        self.best_idx_dir = os.path.join(self.trainer.log_dir, '..', 'best_idx_for_metric')
        os.makedirs(self.best_idx_dir, exist_ok=True)

        if split == "train":
            if self.hparams.stage == "lm_pretrain":
                rs_set = self.train_lm_forward_pretraining(batch, batch_idx)
            elif self.hparams.stage == "lm_instruct":
                rs_set = self.train_lm_forward_instruction_tuning(batch, batch_idx)
            elif self.hparams.stage == "lm_grpo":
                rs_set = self.train_lm_forward_grpo(batch, batch_idx)
            else:
                raise ValueError(f"Unknown stage {self.hparams.stage}")
            return self._losses['losses_' + split].update(rs_set)

        if self.hparams.task == "sceneqa_obstacleqa":
            rs_set = self.val_sceneqa_obstacleqa_forward(batch, batch_idx)
        else:
            rs_set = self.val_motion_and_text_forward(batch, batch_idx)

        lengths = batch['length']
        for metric in self.hparams.metrics_dict:
            if metric == "T2TMetrics":
                self.metrics.T2TMetrics.update(pred_texts=rs_set["t_rst_scene"], gt_texts=rs_set["t_ref_scene"])
            elif metric == "ObstacleMetrics":
                self.metrics.ObstacleMetrics.update(pred_texts=rs_set["t_rst_obstacle"],
                                                    gt_texts=rs_set["t_ref_obstacle"])
            elif metric == "PredMetrics":
                for subtask, sub_rs in rs_set["m2m"].items():
                    lengths_ref = batch["future_motion_length"] if subtask == 'stp2mt_f' else lengths
                    sub_lengths = [int(l) for l in sub_rs["length"]]
                    if not sub_lengths:
                        continue
                    self._get_or_create_subtask_metric("PredMetrics", subtask).update(
                        sub_rs["joints_rst"],
                        sub_rs["joints_ref"],
                        sub_lengths, lengths_ref
                    )
            elif metric == "M2TMetrics":
                for subtask, text_pack in rs_set["m2t"].items():
                    self._get_or_create_subtask_metric("M2TMetrics", subtask).update(
                        pred_texts=text_pack["t_pred"],
                        gt_texts=text_pack["t_ref"],
                        feats_ref=text_pack["feats_ref"],
                        lengths=[int(l) for l in text_pack["length"]],
                        word_embs=batch["word_embs"],
                        pos_ohot=batch["pos_ohot"],
                        text_lengths=batch["text_len"]
                    )
            elif metric == "MotionTextMetrics":
                for subtask, mt_pack in rs_set["motion_text"].items():
                    lengths_int = [int(l) for l in mt_pack["length"]]
                    if not lengths_int:
                        continue
                    self._get_or_create_subtask_metric("MotionTextMetrics", subtask).update(
                        feats_rst=mt_pack["feats_rst"],
                        pred_texts=mt_pack["t_pred"],
                        lengths=lengths_int,
                    )
            else:
                raise TypeError(f"Not support this metric {metric}")
        return None

    def metrics_log_dict(self):
        if self.hparams.task != "motion_narration":
            return super().metrics_log_dict()

        logs = {}
        for metric_name, metric_map in self.subtask_metrics.items():
            for subtask, metric in metric_map.items():
                if hasattr(metric, "count_seq") and getattr(metric, "count_seq").item() == 0:
                    continue
                # call compute() without torchmetrics' wrapper (no cross-process sync / caching)
                compute_fn = getattr(metric.compute, '__wrapped__', None)
                compute = compute_fn.__get__(metric, type(metric)) if compute_fn is not None else metric.compute
                metrics_dict = compute(sanity_flag=self.trainer.sanity_checking, idx_dir=self.best_idx_dir,
                                       subtask=subtask)
                metric.reset()
                for key, value in metrics_dict.items():
                    scalar = value.item() if torch.is_tensor(value) else float(value)
                    logs[f"Metrics/{metric_name}_{subtask}/{key}"] = scalar
        return logs
