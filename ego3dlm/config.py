import glob
import importlib
import os
from argparse import ArgumentParser
from os.path import join as pjoin

from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_module_config(cfg, filepath="./configs"):
    """
    Load the module configs (configs/{lm,vq,evaluator}/*.yaml) as cfg.<folder>.<name>
    """

    yamls = [y for folder in ('lm', 'vq', 'evaluator') for y in glob.glob(pjoin(filepath, folder, '*.yaml'))]
    yamls = [y.replace(filepath, '') for y in yamls]
    for yaml in yamls:
        nodes = yaml.replace('.yaml', '').replace(os.sep, '.')
        nodes = nodes[1:] if nodes[0] == '.' else nodes
        OmegaConf.update(cfg, nodes, OmegaConf.load('./configs' + yaml))

    return cfg


def resume_config(cfg: OmegaConf):
    """
    Resume model and wandb
    """

    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        if os.path.exists(resume):
            # Checkpoints
            cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
            # Wandb
            wandb_files = os.listdir(pjoin(resume, "wandb", "latest-run"))
            wandb_run = [item for item in wandb_files if "run-" in item][0]
            cfg.LOGGER.WANDB.params.id = wandb_run.replace("run-", "").replace(".wandb", "")
        else:
            raise ValueError("Resume path is not right.")

    return cfg


def parse_args(phase="train"):
    """
    Parse arguments and load config files (phase: "train" or "test")
    """

    parser = ArgumentParser()
    group = parser.add_argument_group("Training options")
    group.add_argument("--cfg_assets",
                       type=str,
                       required=False,
                       default="./configs/assets.yaml",
                       help="config file for asset paths")
    group.add_argument("--cfg", type=str, required=True, help="config file")
    group.add_argument("--batch_size", type=int, required=False, help="training batch size")
    group.add_argument("--num_nodes", type=int, required=False, help="number of nodes")
    group.add_argument("--device", type=int, nargs="+", required=False, help="training device")
    group.add_argument("--task", type=str, required=False, help="evaluation task type")
    group.add_argument("--nodebug", action="store_true", required=False, help="debug or not")
    params = parser.parse_args()

    # Load yaml config files
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    # Update config with arguments
    cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
    cfg.DEVICE = params.device if params.device else cfg.DEVICE
    cfg.NUM_NODES = params.num_nodes if params.num_nodes else cfg.NUM_NODES
    if params.task:
        cfg.model.params.task = params.task
    cfg.DEBUG = not params.nodebug

    # Force no debug in test
    if phase == "test":
        cfg.DEBUG = False
        cfg.DEVICE = [0]
        print("Force no debugging and one gpu when testing")

    # Debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    # Resume config
    cfg = resume_config(cfg)

    return cfg
