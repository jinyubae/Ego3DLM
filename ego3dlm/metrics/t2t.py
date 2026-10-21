from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torchmetrics import Metric


class T2TMetrics(Metric):
    """Text-to-text metrics aggregating BLEU, ROUGE-L, and CIDEr."""

    def __init__(
        self,
        cfg,
        bleu_orders: Sequence[int] = (1, 2, 3, 4),
        dist_sync_on_step: bool = True,
        **_: object,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if not bleu_orders:
            raise ValueError("bleu_orders must contain at least one order.")

        self.cfg = cfg
        self.bleu_orders = tuple(sorted(set(int(o) for o in bleu_orders if o >= 1)))
        if not self.bleu_orders:
            raise ValueError("bleu_orders must only contain positive integers.")

        self.metrics = [f"Bleu_{order}" for order in self.bleu_orders]
        self.metrics.extend(["ROUGE_L", "CIDEr", "SBERTScore"])

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_texts", default=[], dist_reduce_fx=None)
        self.add_state("gt_texts", default=[], dist_reduce_fx=None)

        from nlgmetricverse import NLGMetricverse, load_metric  # local import for optional dependency

        metric_defs = [
            load_metric(
                "bleu",
                resulting_name=f"bleu_{order}",
                compute_kwargs={"max_order": order},
            )
            for order in self.bleu_orders
        ]
        metric_defs.append(load_metric("rouge"))
        metric_defs.append(load_metric("cider"))
        self.nlg_evaluator = NLGMetricverse(metric_defs)

        from sentence_transformers import SentenceTransformer
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sbert_model.eval()
        for p in self.sbert_model.parameters():
            p.requires_grad = False

    def _ensure_list(self, value: Iterable[str]) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(v) for v in value]
        raise TypeError("Expected an iterable of strings for text inputs.")

    def _extract_nested(self, value: object, *keys: str) -> object:
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    return value[key]
        return value

    def _to_tensor(self, value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, (int, float)):
            return torch.tensor(float(value), device=self.device)
        if isinstance(value, dict):
            for candidate in ("score", "bleu", "value"):
                if candidate in value:
                    return self._to_tensor(value[candidate])
        return torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def update(self, pred_texts: Sequence[str], gt_texts: Sequence[str]) -> None:
        pred_batch = self._ensure_list(pred_texts)
        gt_batch = self._ensure_list(gt_texts)

        if len(pred_batch) != len(gt_batch):
            raise ValueError("pred_texts and gt_texts must have the same length.")

        self.count += len(pred_batch)
        self.pred_texts.extend(pred_batch)
        self.gt_texts.extend(gt_batch)

    @torch.no_grad()
    def compute(self, sanity_flag: bool = False) -> dict:
        metric_values = {name: torch.tensor(0.0, device=self.device) for name in self.metrics}

        if sanity_flag or self.count.item() == 0 or not self.pred_texts:
            return metric_values

        scores = self.nlg_evaluator(predictions=self.pred_texts, references=self.gt_texts)

        for order in self.bleu_orders:
            metric_key = f"bleu_{order}"
            metric_values[f"Bleu_{order}"] = self._to_tensor(scores.get(metric_key))

        rouge = self._extract_nested(scores.get("rouge"), "rougeL", "rougeLsum", "rougeL_fmeasure")
        cider = self._extract_nested(scores.get("cider"), "score", "cider")

        metric_values["ROUGE_L"] = self._to_tensor(rouge)
        metric_values["CIDEr"] = self._to_tensor(cider)

        pred_embs = self.sbert_model.encode(
            self.pred_texts, convert_to_tensor=True, show_progress_bar=False, device='cpu'
        )
        gt_embs = self.sbert_model.encode(
            self.gt_texts, convert_to_tensor=True, show_progress_bar=False, device='cpu'
        )
        metric_values["SBERTScore"] = torch.tensor(
            F.cosine_similarity(pred_embs, gt_embs).mean().item(), device=self.device
        )

        self.reset()
        return metric_values

    def reset(self) -> None:
        super().reset()
        self.pred_texts = []
        self.gt_texts = []
