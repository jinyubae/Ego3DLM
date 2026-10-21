import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
from ego3dlm.callback import build_callbacks
from ego3dlm.config import instantiate_from_config
from ego3dlm.config import parse_args
from ego3dlm.data.build_data import build_data
from ego3dlm.models.build_model import build_model
from ego3dlm.utils.logger import create_logger
from ego3dlm.utils.load_checkpoint import CheckpointIO, add_special_tokens, load_pretrained


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
    callbacks = build_callbacks(cfg, logger=logger, phase='val')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp" if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        num_sanity_val_steps=2,
        plugins=[CheckpointIO()],
    )
    logger.info("Trainer initialized")

    # The checkpoint already contains the special tokens of its stage; add them before loading
    add_special_tokens(cfg, model)

    assert cfg.TRAIN.PRETRAINED, "Validation requires pretrained checkpoint"
    load_pretrained(cfg, model, logger)

    # When GRPO_VAL is True, weights are already loaded by load_pretrained (with ref_model filtered),
    # so skip ckpt_path to avoid PL re-loading the checkpoint with strict=True.
    ckpt_path = None if cfg.TRAIN.get('GRPO_VAL', False) else cfg.TRAIN.PRETRAINED
    trainer.validate(model,
                     datamodule=datamodule,
                     ckpt_path=ckpt_path)

    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Validation ends!")


if __name__ == "__main__":
    main()
