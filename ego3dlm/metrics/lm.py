import math
import os
from typing import List

import numpy as np
import spacy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics import Metric

from ego3dlm.config import instantiate_from_config
from .obstacle import ObstacleMetrics
from .t2t import T2TMetrics
from .utils import (calculate_activation_statistics_np, calculate_diversity_np, calculate_frechet_distance_np,
                    calculate_top_k, euclidean_distance_matrix)


class LMMetrics(nn.Module):
    """Metrics of the LM stages that are not per subtask: the Stage I QA metrics, and for Stage III the
    text / motion encoders of the GRPO matching reward. The Stage II / III motion and text metrics are created
    per subtask by the model (Ego3DLM._get_or_create_subtask_metric)."""

    def __init__(self, cfg, datamodule, debug, stage, **kwargs) -> None:
        super().__init__()
        if 'T2TMetrics' in cfg.METRIC.TYPE:
            self.T2TMetrics = T2TMetrics(cfg=cfg, bleu_orders=(1, 4))
        if 'ObstacleMetrics' in cfg.METRIC.TYPE:
            self.ObstacleMetrics = ObstacleMetrics()
        if stage == 'lm_grpo':
            self.PredMetrics = PredMetrics(
                cfg=cfg,
                datamodule=datamodule,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
            self.M2TMetrics = M2TMetrics(
                cfg=cfg,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP
            )


def _load_evaluator(metric, cfg):
    """Frozen text / movement / motion encoders of the text-motion matching evaluator (METRIC.FEATURE_EXTRACTOR)."""
    cfg.METRIC.EVALUATOR.movement_encoder.params.input_size = 279 - 4
    metric.text_encoder = instantiate_from_config(cfg.METRIC.EVALUATOR.text_encoder)
    metric.movement_encoder = instantiate_from_config(cfg.METRIC.EVALUATOR.movement_encoder)
    metric.motion_encoder = instantiate_from_config(cfg.METRIC.EVALUATOR.motion_encoder)

    evaluator_checkpoint = torch.load(cfg.METRIC.FEATURE_EXTRACTOR, map_location="cpu")
    metric.text_encoder.load_state_dict(evaluator_checkpoint["text_encoder"])
    metric.movement_encoder.load_state_dict(evaluator_checkpoint["movement_encoder"])
    metric.motion_encoder.load_state_dict(evaluator_checkpoint["motion_encoder"])

    for encoder in (metric.text_encoder, metric.movement_encoder, metric.motion_encoder):
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False


def _gather_list_across_processes(data_list, to_cpu=True):
    if not dist.is_available() or not dist.is_initialized():
        return list(data_list)
    local_list = list(data_list)
    if to_cpu:
        # move tensors to CPU to avoid CUDA OOM during gathering
        local_list = [item.detach().cpu() if isinstance(item, torch.Tensor) else item for item in local_list]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_list)
    merged = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


class _TextEncoderMixin:
    """Text embeddings of the evaluator (needs self.nlp, self.w_vectorizer, self.max_text_len, self.text_encoder)."""

    def _process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def _get_text_embeddings(self, texts):
        word_embs = []
        pos_ohot = []
        text_lengths = []
        for sentence in texts:
            word_list, pos_list = self._process_text(sentence.strip())
            t_tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]

            if len(t_tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = t_tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)

            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(torch.tensor(pos_oh).float()[None])
                word_embeddings.append(torch.tensor(word_emb).float()[None])
            text_lengths.append(sent_len)
            pos_ohot.append(torch.cat(pos_one_hots, dim=0)[None])
            word_embs.append(torch.cat(word_embeddings, dim=0)[None])

        device = self.count.device
        word_embs = torch.cat(word_embs, dim=0).to(device=device, dtype=torch.float32)
        pos_ohot = torch.cat(pos_ohot, dim=0).to(device=device, dtype=torch.float32)
        text_lengths = torch.tensor(text_lengths).to(device=device, dtype=torch.float32)

        align_idx = np.argsort(text_lengths.data.tolist())[::-1].copy()
        text_embeddings = self.text_encoder(word_embs[align_idx], pos_ohot[align_idx], text_lengths[align_idx])

        original_text_embeddings = text_embeddings.clone()
        for idx, sort in enumerate(align_idx):
            original_text_embeddings[sort] = text_embeddings[idx]
        return original_text_embeddings


class M2TMetrics(_TextEncoderMixin, Metric):
    """Narration metrics of a subtask: R-precision / matching score of the evaluator (predicted text vs
    ground-truth motion), BLEU, ROUGE-L, CIDEr and SBERT score, for the best-of-n generation (selected by
    PredMetrics) and for the first generation ('_idx0')."""

    def __init__(self,
                 cfg,
                 w_vectorizer,
                 top_k=3,
                 R_size=32,
                 max_text_len=20,
                 dist_sync_on_step=True,
                 unit_length=4,
                 **kwargs):
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.cfg = cfg
        self.w_vectorizer = w_vectorizer
        self.max_text_len = max_text_len
        self.top_k = top_k
        self.R_size = R_size
        self.unit_length = unit_length
        self.sync_on_compute = False

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # Cached batches
        self.pred_texts = []
        self.gt_texts = []
        self.pred_texts_highest_bleu4 = []
        self.add_state("predtext_embeddings", default=[])
        self.add_state("gttext_embeddings", default=[])
        self.add_state("gtmotion_embeddings", default=[])
        self.add_state("predtext_embeddings_highest_bleu4", default=[])

        _load_evaluator(self, cfg)

        self.nlp = spacy.load('en_core_web_sm')

        from nlgmetricverse import NLGMetricverse, load_metric
        metrics = [
            load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
            load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
            load_metric("rouge"),
            load_metric("cider"),
        ]
        self.nlg_evaluator = NLGMetricverse(metrics)

        from sentence_transformers import SentenceTransformer
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sbert_model.eval()
        for p in self.sbert_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute(self, sanity_flag, idx_dir='', subtask=''):
        num_items = len(self.gt_texts)
        samples_per_item = 1
        best_indices_n = [0 for _ in range(num_items)]
        idx0_indices_n = [0 for _ in range(num_items)]
        use_highest_bleu4 = False
        if idx_dir and subtask:
            # best-of-n sample indices written by PredMetrics of the same subtask
            gpu_idx = self.device.index
            file_path = os.path.join(idx_dir, f'best_ADE_idx_gpu_{subtask}_{gpu_idx}.npy')
            samples_per_item = len(self.pred_texts) // num_items
            if os.path.exists(file_path):
                best_indices_n = np.load(file_path)
                os.remove(file_path)
                assert len(best_indices_n) == num_items
            else:
                use_highest_bleu4 = True

        pred_texts_idx0 = []
        for i in range(num_items):
            pred_texts_idx0.append(self.pred_texts[samples_per_item * i])

        if use_highest_bleu4 and self.pred_texts_highest_bleu4:
            pred_texts_best = list(self.pred_texts_highest_bleu4)
        else:
            pred_texts_best = []
            for i in range(num_items):
                best_idx = samples_per_item * i + int(best_indices_n[i])
                pred_texts_best.append(self.pred_texts[best_idx])

        def _gather_selected_matching_embeddings(local_indices, use_highest_bleu4=False):
            gt_motion_list = []
            gt_text_list = []
            pred_text_list = []
            for i in range(num_items):
                idx_ = samples_per_item * i + int(local_indices[i])
                gt_motion_list.append(self.gtmotion_embeddings[idx_].unsqueeze(0).detach().cpu())
                gt_text_list.append(self.gttext_embeddings[idx_].unsqueeze(0).detach().cpu())
                if use_highest_bleu4:
                    pred_text_list.append(self.predtext_embeddings_highest_bleu4[i].unsqueeze(0).detach().cpu())
                else:
                    pred_text_list.append(self.predtext_embeddings[idx_].unsqueeze(0).detach().cpu())

            gtm_list = _gather_list_across_processes(gt_motion_list)
            gtt_list = _gather_list_across_processes(gt_text_list)
            prt_list = _gather_list_across_processes(pred_text_list)
            if not gtm_list or not gtt_list or not prt_list:
                return None, None, None, 0

            all_motions_emb = torch.cat([t.to(self.device) for t in gtm_list], dim=0)
            all_gttexts_emb = torch.cat([t.to(self.device) for t in gtt_list], dim=0)
            all_predtexts_emb = torch.cat([t.to(self.device) for t in prt_list], dim=0)

            count_seq_local = all_motions_emb.size(0)
            if count_seq_local > 0:
                g = torch.Generator(device=all_motions_emb.device).manual_seed(0)
                shuffle_idx = torch.randperm(count_seq_local, generator=g, device=all_motions_emb.device)
                all_motions_emb = all_motions_emb[shuffle_idx]
                all_gttexts_emb = all_gttexts_emb[shuffle_idx]
                all_predtexts_emb = all_predtexts_emb[shuffle_idx]

            return all_motions_emb, all_gttexts_emb, all_predtexts_emb, count_seq_local

        def _compute_matching_metrics(all_motions_emb, all_gttexts_emb, all_predtexts_emb, count_seq_local, suffix=""):
            if count_seq_local == 0:
                return
            assert count_seq_local >= self.R_size

            top_k_mat = torch.zeros((self.top_k,), device=self.device)
            matching_score_sum = torch.tensor(0.0, device=self.device)
            r_count = (count_seq_local // self.R_size) * self.R_size
            for i in range(count_seq_local // self.R_size):
                s, e = i * self.R_size, (i + 1) * self.R_size
                dist_mat = euclidean_distance_matrix(all_predtexts_emb[s:e], all_motions_emb[s:e]).nan_to_num()
                matching_score_sum += dist_mat.trace()
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(dim=0)
            metrics[f"M2T_Matching_score{suffix}"] = matching_score_sum / r_count
            for k in range(self.top_k):
                metrics[f"M2T_R_precision_top_{k+1}{suffix}"] = top_k_mat[k] / r_count

            top_k_mat = torch.zeros((self.top_k,), device=self.device)
            gt_matching_score_sum = torch.tensor(0.0, device=self.device)
            for i in range(count_seq_local // self.R_size):
                s, e = i * self.R_size, (i + 1) * self.R_size
                dist_mat = euclidean_distance_matrix(all_gttexts_emb[s:e], all_motions_emb[s:e]).nan_to_num()
                gt_matching_score_sum += dist_mat.trace()
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(dim=0)
            metrics[f"M2T_gt_Matching_score{suffix}"] = gt_matching_score_sum / r_count
            for k in range(self.top_k):
                metrics[f"M2T_gt_R_precision_top_{k+1}{suffix}"] = top_k_mat[k] / r_count

        def _compute_nlg_metrics(pred_texts_selected, all_gt_texts, suffix=""):
            all_pred_texts = _gather_list_across_processes(pred_texts_selected)
            if len(all_pred_texts) and len(all_gt_texts):
                # nlgmetricverse may mutate list inputs while filtering empty items.
                scores = self.nlg_evaluator(
                    predictions=list(all_pred_texts),
                    references=list(all_gt_texts),
                )
                bleu1 = scores.get('bleu_1', {}).get('score') or 0.0
                bleu4 = scores.get('bleu_4', {}).get('score') or 0.0
                rouge = scores.get('rouge', {}).get('rougeL') or 0.0
                cider = scores.get('cider', {}).get('score') or 0.0

                # SBERT score: mean cosine similarity between pred and gt embeddings
                pred_embs = self.sbert_model.encode(
                    list(all_pred_texts), convert_to_tensor=True,
                    show_progress_bar=False, device='cpu'
                )
                gt_embs = self.sbert_model.encode(
                    list(all_gt_texts), convert_to_tensor=True,
                    show_progress_bar=False, device='cpu'
                )
                sbert_score = F.cosine_similarity(pred_embs, gt_embs).mean().item()
            else:
                bleu1 = bleu4 = rouge = cider = sbert_score = 0.0
            metrics[f"M2T_Bleu_1{suffix}"] = torch.tensor(bleu1, device=self.device)
            metrics[f"M2T_Bleu_4{suffix}"] = torch.tensor(bleu4, device=self.device)
            metrics[f"M2T_ROUGE_L{suffix}"] = torch.tensor(rouge, device=self.device)
            metrics[f"M2T_CIDEr{suffix}"] = torch.tensor(cider, device=self.device)
            metrics[f"M2T_SBERTScore{suffix}"] = torch.tensor(sbert_score, device=self.device)

        matching_best = _gather_selected_matching_embeddings(best_indices_n, use_highest_bleu4=use_highest_bleu4)
        matching_idx0 = _gather_selected_matching_embeddings(idx0_indices_n)
        self.gtmotion_embeddings = []
        self.gttext_embeddings = []
        self.predtext_embeddings = []
        torch.cuda.empty_cache()

        metrics = {}
        if sanity_flag:
            return metrics

        _compute_matching_metrics(*matching_best, suffix="")
        _compute_matching_metrics(*matching_idx0, suffix="_idx0")

        all_gt_texts = _gather_list_across_processes(self.gt_texts)
        _compute_nlg_metrics(pred_texts_best, all_gt_texts, suffix="")
        _compute_nlg_metrics(pred_texts_idx0, all_gt_texts, suffix="_idx0")

        self.reset(); self.gt_texts = []; self.pred_texts = []; self.pred_texts_highest_bleu4 = []
        return metrics

    @torch.no_grad()
    def update(self,
               pred_texts: List[str],
               gt_texts: List[str],
               feats_ref: Tensor,
               lengths: List[int],
               word_embs: Tensor,
               pos_ohot: Tensor,
               text_lengths: Tensor):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # motion encoder (the ground truth is repeated for the n generations of each sample)
        samples_per_item = 1
        if len(lengths) > feats_ref.shape[0] and len(lengths) % feats_ref.shape[0] == 0:
            samples_per_item = len(lengths) // feats_ref.shape[0]

        feats_ref_expanded = feats_ref.unsqueeze(1).repeat(1, samples_per_item, 1, 1)
        feats_ref_expanded = feats_ref_expanded.reshape(samples_per_item*feats_ref.shape[0], *feats_ref.shape[1:])
        m_lens = torch.tensor(lengths, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref_expanded = feats_ref_expanded[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.UNIT_LEN,
                           rounding_mode="floor")
        ref_mov = self.movement_encoder(feats_ref_expanded[..., :-4]).detach()
        m_lens = m_lens // self.unit_length
        ref_emb = self.motion_encoder(ref_mov, m_lens)
        gtmotion_embeddings = torch.flatten(ref_emb, start_dim=1).detach()
        if not torch.is_tensor(self.gtmotion_embeddings):
            self.gtmotion_embeddings = gtmotion_embeddings
        else:
            self.gtmotion_embeddings = torch.cat((self.gtmotion_embeddings, gtmotion_embeddings), dim=0)

        # text encoder
        word_embs_expanded = word_embs.unsqueeze(1).repeat(1, samples_per_item, 1, 1)
        word_embs_expanded = word_embs_expanded.reshape(samples_per_item*word_embs.shape[0], *word_embs.shape[1:])
        pos_ohot_expanded = pos_ohot.unsqueeze(1).repeat(1, samples_per_item, 1, 1)
        pos_ohot_expanded = pos_ohot_expanded.reshape(samples_per_item*pos_ohot.shape[0], *pos_ohot.shape[1:])
        text_lengths_expanded = text_lengths.unsqueeze(1).repeat(1, samples_per_item)
        text_lengths_expanded = text_lengths_expanded.reshape(-1)

        gttext_emb = self.text_encoder(word_embs_expanded, pos_ohot_expanded,
                                          text_lengths_expanded)[align_idx]
        gttext_embeddings = torch.flatten(gttext_emb, start_dim=1).detach()
        predtext_emb = self._get_text_embeddings(pred_texts)[align_idx]
        predtext_embeddings = torch.flatten(predtext_emb, start_dim=1).detach()

        if not torch.is_tensor(self.gttext_embeddings):
            self.gttext_embeddings = gttext_embeddings
        else:
            self.gttext_embeddings = torch.cat((self.gttext_embeddings, gttext_embeddings), dim=0)

        if not torch.is_tensor(self.predtext_embeddings):
            self.predtext_embeddings = predtext_embeddings
        else:
            self.predtext_embeddings = torch.cat((self.predtext_embeddings, predtext_embeddings), dim=0)

        if samples_per_item > 1:
            # generation with the highest BLEU-4 per sample (used if no best-ADE index is available)
            num_items_batch = feats_ref.shape[0]
            inv_align_idx = np.argsort(align_idx)
            predtext_emb_orig = predtext_emb[inv_align_idx]  # shape: [N*spi, ...]

            bleu4_scores = []
            for k in range(len(pred_texts)):
                pred = pred_texts[k]
                ref = gt_texts[k // samples_per_item]
                if not pred or not ref or not pred.strip() or not ref.strip():
                    bleu4_scores.append(0.0)
                else:
                    bleu4_scores.append(
                        (self.nlg_evaluator(
                            predictions=[pred],
                            references=[ref]
                        ).get('bleu_4', {}).get('score') or 0.0)
                    )
            bleu4_arr = np.array(bleu4_scores).reshape(num_items_batch, samples_per_item)
            best_j_per_item = np.argmax(bleu4_arr, axis=1)

            best_embs = []
            for i in range(num_items_batch):
                best_idx = i * samples_per_item + int(best_j_per_item[i])
                best_flat = torch.flatten(
                    predtext_emb_orig[best_idx].unsqueeze(0), start_dim=1
                ).detach()
                best_embs.append(best_flat)
                self.pred_texts_highest_bleu4.append(pred_texts[best_idx])
            best_embs_tensor = torch.cat(best_embs, dim=0)
            if not torch.is_tensor(self.predtext_embeddings_highest_bleu4):
                self.predtext_embeddings_highest_bleu4 = best_embs_tensor
            else:
                self.predtext_embeddings_highest_bleu4 = torch.cat(
                    (self.predtext_embeddings_highest_bleu4, best_embs_tensor), dim=0
                )

        self.pred_texts.extend(pred_texts)
        self.gt_texts.extend(gt_texts)


class MotionTextMetric(_TextEncoderMixin, Metric):
    """
    Computes matching score between predicted motion and predicted text.
    Unlike M2TMetrics (which compares pred_text vs gt_motion),
    this metric evaluates self-consistency of joint motion+text predictions.
    """

    def __init__(self,
                 cfg,
                 w_vectorizer,
                 R_size=32,
                 max_text_len=20,
                 dist_sync_on_step=True,
                 unit_length=4,
                 **kwargs):
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.cfg = cfg
        self.w_vectorizer = w_vectorizer
        self.max_text_len = max_text_len
        self.R_size = R_size
        self.unit_length = unit_length
        self.sync_on_compute = False

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # Cached embeddings (stored per-sample, not batched)
        self.add_state("predmotion_embeddings", default=[])
        self.add_state("predtext_embeddings", default=[])

        _load_evaluator(self, cfg)

        self.nlp = spacy.load('en_core_web_sm')

    @torch.no_grad()
    def update(self,
               feats_rst: Tensor,
               pred_texts: List[str],
               lengths: List[int]):
        """
        Args:
            feats_rst: predicted motion features [B, T, D] (renormed for evaluator)
            pred_texts: list of predicted text strings, length B
            lengths: list of motion frame lengths, length B
        """
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # Encode predicted motion
        m_lens = torch.tensor(lengths, device=feats_rst.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_rst_sorted = feats_rst[align_idx]
        m_lens_sorted = m_lens[align_idx]
        m_lens_sorted = torch.div(m_lens_sorted, self.cfg.DATASET.UNIT_LEN, rounding_mode="floor")
        rst_mov = self.movement_encoder(feats_rst_sorted[..., :-4]).detach()
        m_lens_sorted = m_lens_sorted // self.unit_length
        rst_emb = self.motion_encoder(rst_mov, m_lens_sorted)
        predmotion_embeddings_sorted = torch.flatten(rst_emb, start_dim=1).detach()

        # Restore original order
        predmotion_embeddings = torch.zeros_like(predmotion_embeddings_sorted)
        for idx, sort in enumerate(align_idx):
            predmotion_embeddings[sort] = predmotion_embeddings_sorted[idx]

        # Encode predicted text
        predtext_emb = self._get_text_embeddings(pred_texts)
        predtext_embeddings = torch.flatten(predtext_emb, start_dim=1).detach()

        # Store per-sample
        for i in range(len(lengths)):
            self.predmotion_embeddings.append(predmotion_embeddings[i].unsqueeze(0).detach().cpu())
            self.predtext_embeddings.append(predtext_embeddings[i].unsqueeze(0).detach().cpu())

    @torch.no_grad()
    def compute(self, sanity_flag, idx_dir='', subtask=''):
        num_items = self.count_seq.item()
        samples_per_item = 1
        best_indices_n = [0 for _ in range(num_items)]

        if idx_dir and subtask:
            # best-of-n sample indices written by PredMetrics of the same subtask
            gpu_idx = self.device.index
            file_path = os.path.join(idx_dir, f'best_ADE_idx_gpu_{subtask}_{gpu_idx}.npy')
            if os.path.exists(file_path):
                best_indices_n = np.load(file_path)
                samples_per_item = len(self.predmotion_embeddings) // len(best_indices_n)

        metrics = {"Matching_score": torch.tensor(0.0, device=self.device)}
        if sanity_flag:
            return metrics

        # Select best sample per item
        selected_motion_list = []
        selected_text_list = []
        for i in range(len(best_indices_n)):
            best_idx = samples_per_item * i + int(best_indices_n[i])
            selected_motion_list.append(self.predmotion_embeddings[best_idx])
            selected_text_list.append(self.predtext_embeddings[best_idx])

        # Gather across processes
        all_motion_list = _gather_list_across_processes(selected_motion_list)
        all_text_list = _gather_list_across_processes(selected_text_list)
        if not all_motion_list or not all_text_list:
            return metrics

        all_motions = torch.cat([t.to(self.device) for t in all_motion_list], dim=0)
        all_texts = torch.cat([t.to(self.device) for t in all_text_list], dim=0)

        count_seq_total = all_motions.size(0)
        if count_seq_total < self.R_size:
            return metrics

        # Shuffle with fixed seed for reproducibility
        g = torch.Generator(device=all_motions.device).manual_seed(0)
        shuffle_idx = torch.randperm(count_seq_total, generator=g, device=all_motions.device)
        all_motions = all_motions[shuffle_idx]
        all_texts = all_texts[shuffle_idx]

        # Matching score (pred_text vs pred_motion)
        matching_score_sum = torch.tensor(0.0, device=self.device)
        r_count = (count_seq_total // self.R_size) * self.R_size
        for i in range(count_seq_total // self.R_size):
            s, e = i * self.R_size, (i + 1) * self.R_size
            dist_mat = euclidean_distance_matrix(all_texts[s:e], all_motions[s:e]).nan_to_num()
            matching_score_sum += dist_mat.trace()
        metrics["Matching_score"] = matching_score_sum / r_count

        self.reset()
        return metrics


class PredMetrics(Metric):
    """Motion metrics of a subtask for the best of the n generations (lowest mean joint error):
    prediction ('_f', is_future) APE / JPE / head ADE / FDE (+ 2 s and first-generation '_idx0' variants),
    tracking ('_c') APE / upper / lower body error and root / joint angle errors, plus FID and Diversity
    of the evaluator motion embeddings. The selected indices are saved for M2TMetrics / MotionTextMetric."""

    def __init__(self,
                 cfg,
                 datamodule,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 is_future: bool = False,
                 **kwargs):
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.joints2feats = datamodule.joints2feats
        self.cfg = cfg
        self.diversity_times = diversity_times
        self.sync_on_compute = False
        self.is_future = is_future
        # Joint groups for upper/lower ADE (23-joint xsens skeleton)
        self.upper_joint_ids = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.lower_joint_ids = [0, 1, 15, 16, 17, 18, 19, 20, 21, 22]
        # Parent-child bone pairs derived from KINEMATIC_CHAIN
        self.bone_pairs = [
            (0, 1), (1, 15), (15, 16), (16, 17), (17, 18),   # left leg
            (1, 19), (19, 20), (20, 21), (21, 22),             # right leg
            (0, 2), (2, 3), (3, 4), (4, 5), (5, 6),           # spine -> head
            (4, 7), (7, 8), (8, 9), (9, 10),                   # left arm
            (4, 11), (11, 12), (12, 13), (13, 14),             # right arm
        ]

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.valid_indices = []

        self.add_state("ADE_local", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if self.is_future:
            self.add_state("ADE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state("FDE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state("ADE_head", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ADE_head_2s", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("FDE_2s", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("FDE_head", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ADE_idx0", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state("FDE_idx0", default=torch.tensor([0.0]), dist_reduce_fx="sum")
            self.add_state("FDE_head_idx0", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ADE_head_idx0", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ADE_head_2s_idx0", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("FDE_2s_idx0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("ADE_upper", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ADE_lower", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("RootAngle", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("JointAngle", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Cached evaluator embeddings for FID / Diversity
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)
        _load_evaluator(self, cfg)

    def compute(self, sanity_flag, idx_dir='', subtask=''):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_counts = [torch.zeros_like(self.count) for _ in range(world_size)]
            gathered_count_seq = [torch.zeros_like(self.count_seq) for _ in range(world_size)]
            dist.all_gather(gathered_counts, self.count)
            dist.all_gather(gathered_count_seq, self.count_seq)
            total_count = torch.stack(gathered_counts).sum()
            total_count_seq = torch.stack(gathered_count_seq).sum()
        else:
            total_count = self.count
            total_count_seq = self.count_seq

        if idx_dir and subtask:
            # save the selected generation per sample for the text metrics of this subtask
            assert len(self.valid_indices) == self.count_seq
            os.makedirs(idx_dir, exist_ok=True)
            gpu_idx = self.device.index
            np.save(os.path.join(idx_dir, f'best_ADE_idx_gpu_{subtask}_{gpu_idx}.npy'), np.array(self.valid_indices))
            self.valid_indices = []

        metrics = {"Avg_seq_len": total_count.to(torch.float32) / total_count_seq.to(torch.float32)}

        if self.is_future:
            names = ["ADE_head", "ADE_head_2s", "FDE_2s", "FDE_head", "ADE_local", "ADE", "FDE",
                     "ADE_head_idx0", "ADE_head_2s_idx0", "FDE_2s_idx0", "FDE_head_idx0", "ADE_idx0", "FDE_idx0"]
        else:
            names = ["ADE_local", "ADE_upper", "ADE_lower", "RootAngle", "JointAngle"]
        sums = [getattr(self, name).view(-1).sum() for name in names]
        if dist.is_available() and dist.is_initialized():
            for s in sums:
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for name, s in zip(names, sums):
            metrics[name] = s / total_count_seq

        # FID and Diversity of the evaluator embeddings
        recmotions = _gather_list_across_processes(self.recmotion_embeddings, to_cpu=False)
        gtmotions = _gather_list_across_processes(self.gtmotion_embeddings, to_cpu=False)
        if recmotions and gtmotions:
            all_genmotions = torch.cat([tensor.detach().cpu() for tensor in recmotions], axis=0)
            all_gtmotions = torch.cat([tensor.detach().cpu() for tensor in gtmotions], axis=0)
            perm_gen = torch.randperm(all_genmotions.shape[0])
            perm_gt = torch.randperm(all_gtmotions.shape[0])
            all_genmotions_np = all_genmotions[perm_gen].numpy()
            all_gtmotions_np = all_gtmotions[perm_gt].numpy()
            if not np.isfinite(all_genmotions_np).all() or not np.isfinite(all_gtmotions_np).all():
                raise ValueError('Non-finite motion embeddings detected before FID computation in PredMetrics.compute')
            mu, cov = calculate_activation_statistics_np(all_genmotions_np)
            gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions_np)
            if self.is_future:
                metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
                diversity_times = min(self.diversity_times, all_genmotions.shape[0] - 1)
                if diversity_times > 0:
                    metrics["Diversity"] = calculate_diversity_np(all_genmotions_np, diversity_times)
                    metrics["gt_Diversity"] = calculate_diversity_np(all_gtmotions_np, diversity_times)
                else:
                    metrics["Diversity"] = torch.tensor(0.0)
                    metrics["gt_Diversity"] = torch.tensor(0.0)
            else:
                metrics["FID_local"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
                # Diversity is not reported for tracking; it is still drawn so that the random state
                # (and with it the Diversity of the prediction subtask) stays the same
                diversity_times = min(self.diversity_times, all_genmotions.shape[0] - 1)
                if diversity_times > 0:
                    calculate_diversity_np(all_genmotions_np, diversity_times)
                    calculate_diversity_np(all_gtmotions_np, diversity_times)
        else:
            metrics["FID"] = torch.tensor(float('inf'), device=self.device)
            if self.is_future:
                metrics["Diversity"] = torch.tensor(0.0, device=self.device)
                metrics["gt_Diversity"] = torch.tensor(0.0, device=self.device)

        self.reset()

        for key, value in metrics.items():
            if not torch.is_tensor(value):
                metrics[key] = torch.tensor(value, device=self.device)
        return metrics

    def update(self, joints_rst: Tensor, joints_ref: Tensor,
               lengths_rst: List[int], lengths_ref: List[int]):
        """
        joints_rst: (B * n, T, J, 3) the n generations of each sample, joints_ref: (B, T, J, 3)
        """
        assert joints_rst.dim() == 4

        def _to_int_list(values):
            if isinstance(values, torch.Tensor):
                return [int(v) for v in values.tolist()]
            return [int(v) for v in values]

        lengths_ref_list = _to_int_list(lengths_ref)
        lengths_rst_list = _to_int_list(lengths_rst)

        self.count += sum(lengths_ref_list)
        self.count_seq += len(lengths_ref_list)

        num_refs = len(lengths_ref_list)
        if num_refs == 0:
            return

        total_preds = joints_rst.shape[0]
        samples_per_item = 1
        if total_preds >= num_refs and total_preds % num_refs == 0:
            samples_per_item = total_preds // num_refs

        # pad every generation to the reference length with its last frame
        padded_rst = []
        max_ref_len = joints_ref.shape[1]
        pred_max_len = joints_rst.shape[1]
        for idx in range(total_preds):
            seq_len = max(int(lengths_rst_list[idx]), 1)
            last_idx = min(seq_len, pred_max_len) - 1
            last_frame = joints_rst[idx, last_idx, ...]
            padded = last_frame.expand(max_ref_len, -1, 3).clone().to(joints_ref.device)
            effective_len = min(seq_len, max_ref_len, pred_max_len)
            if effective_len > 0:
                padded[:effective_len] = joints_rst[idx, :effective_len]
            padded_rst.append(padded.unsqueeze(0))
        rst = torch.cat(padded_rst, dim=0)
        ref = joints_ref
        B = ref.shape[0]

        # "local": the generation translated so that its head follows the ground-truth head
        ref_expanded = ref.unsqueeze(1).repeat(1, samples_per_item, *([1] * (ref.dim() - 1)))
        ref_expanded = ref_expanded.reshape(samples_per_item*B, *ref_expanded.shape[2:])
        per_joint_local_error = rst[..., 6, :] - ref_expanded[..., 6, :]
        per_joint_local_error = per_joint_local_error[:, :, None].repeat(1, 1, 23, 1)
        rst_local = rst.clone() - per_joint_local_error

        for i, L in enumerate(lengths_ref_list):
            if L <= 0:
                continue

            start_idx = i * samples_per_item
            end_idx = min(start_idx + samples_per_item, total_preds)
            candidate_indices = list(range(start_idx, end_idx))
            if not candidate_indices:
                continue

            rsts = rst[candidate_indices, :L]
            rsts_local = rst_local[candidate_indices, :L]
            refi = ref[i, :L]

            # best of the n generations: lowest mean joint error
            diffs = rsts - refi[None]
            dists = diffs.norm(dim=-1).mean(dim=-1).mean(dim=-1)
            if torch.isnan(dists).any():
                dists = torch.nan_to_num(dists, nan=float('inf'))

            best_local_idx = int(dists.argmin().item())
            diff = diffs[best_local_idx]
            diff_idx0 = diffs[0]
            diff_local = (rsts_local - refi[None])[best_local_idx]
            self.valid_indices.append(best_local_idx)

            dist = torch.linalg.norm(diff, dim=-1)[None]
            dist_local = torch.linalg.norm(diff_local, dim=-1)[None]
            dist_head = dist[:, :, 6]
            dist_head_2s = dist
            if dist_head.shape[1] > 20:
                dist_head_2s = dist[:, :20]

            ade = dist.mean()
            ade_local = dist_local.mean()
            ade_head = dist_head.mean()
            ade_head_2s = dist_head_2s.mean()

            fde_2s = dist[:, -1].mean()
            if dist.shape[1] > 20:
                fde_2s = dist[:, 20-1].mean()
            fde = dist[:, -1].mean()
            fde_head = dist[:, -1, 6]

            if self.is_future:
                dist_idx0 = torch.linalg.norm(diff_idx0, dim=-1)[None]
                dist_head_idx0 = dist_idx0[:, :, 6]
                dist_head_2s_idx0 = dist_idx0
                if dist_head_idx0.shape[1] > 20:
                    dist_head_2s_idx0 = dist_idx0[:, :20]

                ade_idx0 = dist_idx0.mean()
                ade_head_idx0 = dist_head_idx0.mean()
                ade_head_2s_idx0 = dist_head_2s_idx0.mean()

                fde_2s_idx0 = dist_idx0[:, -1].mean()
                if dist_idx0.shape[1] > 20:
                    fde_2s_idx0 = dist_idx0[:, 20-1].mean()
                fde_idx0 = dist_idx0[:, -1].mean()
                fde_head_idx0 = dist_idx0[:, -1, 6]

                self.ADE = self.ADE + ade
                self.FDE = self.FDE + fde
                self.FDE_head = self.FDE_head + fde_head
                self.ADE_head = self.ADE_head + ade_head
                self.ADE_head_2s = self.ADE_head_2s + ade_head_2s
                self.FDE_2s = self.FDE_2s + fde_2s
                self.ADE_idx0 = self.ADE_idx0 + ade_idx0
                self.FDE_idx0 = self.FDE_idx0 + fde_idx0
                self.FDE_head_idx0 = self.FDE_head_idx0 + fde_head_idx0
                self.ADE_head_idx0 = self.ADE_head_idx0 + ade_head_idx0
                self.ADE_head_2s_idx0 = self.ADE_head_2s_idx0 + ade_head_2s_idx0
                self.FDE_2s_idx0 = self.FDE_2s_idx0 + fde_2s_idx0

            self.ADE_local = self.ADE_local + ade_local

            if not self.is_future:
                # ADE upper / lower on the local (head-aligned) generation
                ade_upper = diff_local[:, self.upper_joint_ids].norm(dim=-1).mean()
                ade_lower = diff_local[:, self.lower_joint_ids].norm(dim=-1).mean()

                # Rotation angle metrics using joint positions
                pred_seg = rsts_local[best_local_idx]  # (T, J, 3)
                gt_seg = refi                          # (T, J, 3)

                R_pred_root = self._build_root_rotation(pred_seg)
                R_gt_root = self._build_root_rotation(gt_seg)
                root_angle = self._geodesic_error_deg(R_pred_root, R_gt_root)

                joint_angle_errors = []
                for parent_idx, child_idx in self.bone_pairs:
                    R_pred_bone = self._build_bone_rotation(pred_seg, parent_idx, child_idx)
                    R_gt_bone = self._build_bone_rotation(gt_seg, parent_idx, child_idx)
                    joint_angle_errors.append(self._geodesic_error_deg(R_pred_bone, R_gt_bone))
                joint_angle = torch.stack(joint_angle_errors).mean()

                self.ADE_upper = self.ADE_upper + ade_upper
                self.ADE_lower = self.ADE_lower + ade_lower
                self.RootAngle = self.RootAngle + root_angle
                self.JointAngle = self.JointAngle + joint_angle

        # Evaluator embeddings of the ground truth and of all generations (local ones for tracking)
        if not self.is_future:
            joints_rst = rst_local
        feats_ref = self.get_motion_rep_nymeria(joints_ref)
        feats_rst_all = self.get_motion_rep_nymeria(joints_rst)
        if not torch.isfinite(feats_ref).all():
            raise ValueError('Non-finite value found in feats_ref during PredMetrics.update')
        if not torch.isfinite(feats_rst_all).all():
            raise ValueError('Non-finite value found in feats_rst during PredMetrics.update')

        align_idx_ref = np.argsort(lengths_ref_list)[::-1].copy()
        gtmotion_embeddings = self.get_motion_embeddings(
            feats_ref[align_idx_ref], np.array(lengths_ref_list)[align_idx_ref])
        cache = [0] * len(lengths_ref_list)
        for i in range(len(lengths_ref_list)):
            cache[align_idx_ref[i]] = gtmotion_embeddings[i:i + 1]
        self.gtmotion_embeddings.extend(cache)

        align_idx_rst = np.argsort(lengths_rst_list)[::-1].copy()
        recmotion_embeddings = self.get_motion_embeddings(
            feats_rst_all[align_idx_rst], np.array(lengths_rst_list)[align_idx_rst])
        cache = [0] * len(lengths_rst_list)
        for i in range(len(lengths_rst_list)):
            cache[align_idx_rst[i]] = recmotion_embeddings[i:i + 1]
        self.recmotion_embeddings.extend(cache)

    def _build_root_rotation(self, joints: Tensor) -> Tensor:
        """Build rotation matrices for root joint from joint positions.

        Args:
            joints: (T, J, 3)
        Returns:
            R: (T, 3, 3) — orthonormal frame at root
        """
        up = joints[:, 2] - joints[:, 0]     # spine direction  (T, 3)
        right = joints[:, 1] - joints[:, 0]  # hip direction    (T, 3)
        up = F.normalize(up, dim=-1)
        right = right - (right * up).sum(-1, keepdim=True) * up
        right = F.normalize(right, dim=-1)
        forward = torch.linalg.cross(up, right)
        forward = F.normalize(forward, dim=-1)
        return torch.stack([right, forward, up], dim=-1)  # (T, 3, 3)

    def _build_bone_rotation(self, joints: Tensor, parent_idx: int, child_idx: int) -> Tensor:
        """Build rotation matrices for a bone from joint positions.

        Args:
            joints: (T, J, 3)
            parent_idx, child_idx: bone joint indices
        Returns:
            R: (T, 3, 3)
        """
        bone = joints[:, child_idx] - joints[:, parent_idx]  # (T, 3)
        bone = F.normalize(bone, dim=-1)
        ref = torch.zeros_like(bone)
        ref[:, 1] = 1.0  # world-up
        dot = (bone * ref).sum(-1, keepdim=True).abs()
        ref_alt = torch.zeros_like(bone)
        ref_alt[:, 0] = 1.0  # world-right fallback when bone ~ up
        ref = torch.where(dot > 0.9, ref_alt, ref)
        perp = ref - (ref * bone).sum(-1, keepdim=True) * bone
        perp = F.normalize(perp, dim=-1)
        binorm = torch.linalg.cross(bone, perp)
        return torch.stack([bone, perp, binorm], dim=-1)  # (T, 3, 3)

    def _geodesic_error_deg(self, R_pred: Tensor, R_gt: Tensor) -> Tensor:
        """Mean geodesic angular error in degrees.

        Args:
            R_pred, R_gt: (..., 3, 3)
        Returns:
            scalar tensor (degrees)
        """
        R_rel = R_pred.transpose(-1, -2) @ R_gt
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.acos(cos_theta).mean() * (180.0 / math.pi)

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        m_lens = torch.tensor(lengths)
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.UNIT_LEN,
                           rounding_mode="floor")
        m_lens = m_lens // self.cfg.DATASET.UNIT_LEN
        mov = self.movement_encoder(feats[..., :-4]).detach()
        emb = self.motion_encoder(mov, m_lens)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        return torch.flatten(emb, start_dim=1).detach()

    def get_motion_rep_nymeria(self, joints):
        reps = []
        for i in range(len(joints)):
            rep = self.joints2feats(joints[i].cpu().numpy())
            reps.append(torch.from_numpy(rep).float().to(joints.device))
        reps = torch.stack(reps)
        return torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
