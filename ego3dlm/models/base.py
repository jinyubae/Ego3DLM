import torch
from pytorch_lightning import LightningModule
from collections import OrderedDict
from ego3dlm.metrics import BaseMetrics
from ego3dlm.config import get_obj_from_str


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configure_metrics()

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)


    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)


    def on_train_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        dico.update(self.loss_log_dict('val'))
        # Log metrics
        dico.update(self.metrics_log_dict())
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)


    def preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        
        metric_state_dict = self.metrics.state_dict()
        loss_state_dict = self._losses.state_dict()

        for k, v in metric_state_dict.items():
            new_state_dict['metrics.' + k] = v

        for k, v in loss_state_dict.items():
            new_state_dict['_losses.' + k] = v

        for k, v in state_dict.items():
            if '_losses' not in k and 'Metrics' not in k:
                new_state_dict[k] = v

        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = self.preprocess_state_dict(state_dict)
        super().load_state_dict(new_state_dict, strict)

    def step_log_dict(self):
        return {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.global_step)
        }

    def loss_log_dict(self, split: str):
        losses = self._losses['losses_' + split]
        loss_dict = losses.compute(split)
        return loss_dict

    def metrics_log_dict(self):
        # Compute all metrics
        metrics_log_dict = {}
        for metric in self.hparams.metrics_dict:
            metrics_dict = getattr(
                self.metrics,
                metric).compute(sanity_flag=self.trainer.sanity_checking)
            metrics_log_dict.update({
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            })
        return metrics_log_dict
    
    def configure_optimizers(self):
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(
            params=self.parameters(), **self.hparams.cfg.TRAIN.OPTIM.params)

        # Scheduler
        sched_cfg = self.hparams.cfg.TRAIN.LR_SCHEDULER
        scheduler_target = sched_cfg.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        main_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **sched_cfg.params)

        warmup_epochs = sched_cfg.get('warmup_epochs', 0)
        if warmup_epochs and warmup_epochs > 0:
            warmup_start_factor = sched_cfg.get('warmup_start_factor', 0.01)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            lr_scheduler = main_scheduler

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

