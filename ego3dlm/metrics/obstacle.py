import re
import torch
from torchmetrics import Metric
from typing import List


def _parse_obstacle_fields(text: str):
    """Parse an obstacle answer string into per-direction collision and free-space fields.

    Expected format (as produced by _parse_obstacle_label_to_text):
        "Front: blocked (LOW clearance). Left: clear (MID clearance). Right: clear (HIGH clearance). Best direction: front."

    Returns:
        dict: {direction: {'collision': 'blocked'|'clear'|None, 'free': 'LOW'|'MID'|'HIGH'|None}}
    """
    result = {}
    for direction in ['Front', 'Left', 'Right']:
        m = re.search(
            rf'{direction}:\s*(blocked|clear)\s*\((\w+)\s+clearance\)',
            text, re.IGNORECASE
        )
        if m:
            collision = m.group(1).lower()   # 'blocked' or 'clear'
            free_level = m.group(2).upper()  # 'LOW', 'MID', or 'HIGH'
        else:
            collision = None
            free_level = None
        result[direction.lower()] = {'collision': collision, 'free': free_level}
    return result


class ObstacleMetrics(Metric):
    """Per-direction collision accuracy (blocked/clear) and free-space level accuracy (LOW/MID/HIGH)."""

    DIRECTIONS = ['front', 'left', 'right']

    def __init__(self, dist_sync_on_step: bool = True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.metrics = []
        for d in self.DIRECTIONS:
            self.add_state(f'collision_{d}_correct', default=torch.tensor(0), dist_reduce_fx='sum')
            self.add_state(f'collision_{d}_total',   default=torch.tensor(0), dist_reduce_fx='sum')
            self.add_state(f'free_{d}_correct',      default=torch.tensor(0), dist_reduce_fx='sum')
            self.add_state(f'free_{d}_total',        default=torch.tensor(0), dist_reduce_fx='sum')
            self.metrics += [f'Collision_Acc_{d}', f'FreeSpace_Acc_{d}']
        self.metrics += ['Collision_Acc_mean', 'FreeSpace_Acc_mean']

    @torch.no_grad()
    def update(self, pred_texts: List[str], gt_texts: List[str]) -> None:
        for pred, gt in zip(pred_texts, gt_texts):
            pred_parsed = _parse_obstacle_fields(pred)
            gt_parsed   = _parse_obstacle_fields(gt)
            for d in self.DIRECTIONS:
                gt_col   = gt_parsed[d]['collision']
                pred_col = pred_parsed[d]['collision']
                if gt_col is not None and pred_col is not None:
                    getattr(self, f'collision_{d}_total').add_(1)
                    if gt_col == pred_col:
                        getattr(self, f'collision_{d}_correct').add_(1)

                gt_free   = gt_parsed[d]['free']
                pred_free = pred_parsed[d]['free']
                if gt_free is not None and pred_free is not None:
                    getattr(self, f'free_{d}_total').add_(1)
                    if gt_free == pred_free:
                        getattr(self, f'free_{d}_correct').add_(1)

    @torch.no_grad()
    def compute(self, sanity_flag: bool = False) -> dict:
        if sanity_flag:
            return {k: torch.tensor(0.0) for k in self.metrics}

        metrics = {}
        collision_accs = []
        free_accs = []
        for d in self.DIRECTIONS:
            col_total    = getattr(self, f'collision_{d}_total').item()
            col_correct  = getattr(self, f'collision_{d}_correct').item()
            free_total   = getattr(self, f'free_{d}_total').item()
            free_correct = getattr(self, f'free_{d}_correct').item()

            col_acc  = torch.tensor(col_correct  / col_total  if col_total  > 0 else 0.0)
            free_acc = torch.tensor(free_correct / free_total if free_total > 0 else 0.0)

            metrics[f'Collision_Acc_{d}']  = col_acc
            metrics[f'FreeSpace_Acc_{d}'] = free_acc
            collision_accs.append(col_acc)
            free_accs.append(free_acc)

        metrics['Collision_Acc_mean'] = torch.stack(collision_accs).mean()
        metrics['FreeSpace_Acc_mean'] = torch.stack(free_accs).mean()
        return metrics
