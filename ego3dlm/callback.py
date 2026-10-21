import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint

# Keep the checkpoint with the best value of these metrics: METRIC.TYPE entry -> (logged key, abbreviation, mode)
METRIC_MONITORS = {
    'MRMetrics': ('Metrics/MPJPE', 'MPJPE', 'min'),                              # Stage 0
    'T2TMetrics': ('Metrics/Bleu_1', 'Bleu_1', 'max'),                           # Stage I
    'ObstacleMetrics': ('Metrics/FreeSpace_Acc_mean', 'Acc', 'max'),             # Stage I
    'PredMetrics': ('Metrics/PredMetrics_stp2mt_f/ADE_head', 'ADE_head', 'min'),  # Stages II / III
}


def build_callbacks(cfg, logger=None, phase='test', periodic_checkpoints=True):
    callbacks = [ProgressBar()]
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, periodic_checkpoints=periodic_checkpoints))
    return callbacks


def getCheckpointCallback(cfg, logger=None, periodic_checkpoints=True):
    callbacks = [ProgressLogger(logger, metric_monitor={"loss_total": "total/train"}, log_every_n_steps=1)]
    dirpath = os.path.join(cfg.FOLDER_EXP, "checkpoints")

    ckpt_every_epochs = cfg.LOGGER.get('CKPT_EVERY_EPOCHS', False)
    if ckpt_every_epochs:
        # every CKPT_EVERY_EPOCHS epochs, all kept
        callbacks.append(ModelCheckpoint(dirpath=dirpath, filename="{epoch}", monitor="total/train", mode="min",
                                         every_n_epochs=ckpt_every_epochs, save_top_k=-1, save_last=True,
                                         save_on_train_epoch_end=True))
    else:
        # the latest SAVE_TOP_K checkpoints, and (periodic_checkpoints) one every 10 validation intervals
        callbacks.append(ModelCheckpoint(dirpath=dirpath, filename="{epoch}", monitor="step", mode="max",
                                         every_n_epochs=cfg.LOGGER.VAL_EVERY_STEPS, save_top_k=cfg.LOGGER.SAVE_TOP_K,
                                         save_last=True, save_on_train_epoch_end=True))
        if periodic_checkpoints:
            callbacks.append(ModelCheckpoint(dirpath=dirpath, filename="{epoch}", monitor="step", mode="max",
                                             every_n_epochs=cfg.LOGGER.VAL_EVERY_STEPS * 10, save_top_k=-1,
                                             save_last=False, save_on_train_epoch_end=True))

    # best value of the monitored metrics
    for metric in cfg.METRIC.TYPE:
        if metric in METRIC_MONITORS:
            key, abbr, mode = METRIC_MONITORS[metric]
            callbacks.append(ModelCheckpoint(dirpath=dirpath, filename=f"{mode}-{abbr}" + "-{epoch}-{step}",
                                             monitor=key, mode=mode, every_n_epochs=1, save_top_k=1,
                                             save_last=bool(ckpt_every_epochs), save_on_train_epoch_end=False))
    return callbacks


class ProgressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class ProgressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)
