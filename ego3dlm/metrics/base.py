from torch import nn
from .mr import MRMetrics


class BaseMetrics(nn.Module):
    """Metrics of the motion tokenizer (Stage 0): reconstruction errors."""

    def __init__(self, cfg, datamodule, **kwargs) -> None:
        super().__init__()
        self.MRMetrics = MRMetrics(
            njoints=datamodule.njoints,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
        )
