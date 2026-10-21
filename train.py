import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
from ego3dlm.callback import build_callbacks
from ego3dlm.config import instantiate_from_config
from ego3dlm.config import parse_args
from ego3dlm.data.build_data import build_data
from ego3dlm.models.build_model import build_model
from ego3dlm.utils.logger import create_logger
from ego3dlm.utils.load_checkpoint import (CheckpointIO, add_special_tokens, load_pretrained, load_pretrained_vae,
                                           load_pretrained_grpo_ref)


def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tensorboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train', periodic_checkpoints=False)
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    val_interval = cfg.LOGGER.get('VAL_INTERVAL', None)
    if val_interval is not None:
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    if len(cfg.DEVICE) > 1:
        strategy = cfg.TRAIN.get('STRATEGY', 'ddp_find_unused_parameters_true')
    else:
        strategy = 'auto'

    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.TRAIN.get('PRECISION', 32),
        logger=pl_loggers,
        callbacks=callbacks,
        val_check_interval=val_interval,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy=strategy,
        benchmark=False,
        deterministic=False,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.TRAIN.GRADIENT_ACCUMULATION_STEP,
        plugins=[CheckpointIO()],
    )
    logger.info("Trainer initialized")

    # A checkpoint that already contains the Stage II / III special tokens (INIT_SPECIAL_TOKENS) needs them
    # before loading; otherwise they are added to the loaded previous-stage model
    if cfg.TRAIN.get("INIT_SPECIAL_TOKENS", False):
        add_special_tokens(cfg, model)

    # Load the previous-stage model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger)

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Load GRPO reference model
    if cfg.TRAIN.get("GRPO") and cfg.TRAIN.GRPO.get("REF_MODEL_PATH"):
        load_pretrained_grpo_ref(cfg, model, logger)

    if not cfg.TRAIN.get("INIT_SPECIAL_TOKENS", False):
        add_special_tokens(cfg, model)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
