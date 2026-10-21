from omegaconf import OmegaConf
from ego3dlm.config import instantiate_from_config


def build_data(cfg, phase="train"):
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    data_config['params'] = {'cfg': cfg, 'phase': phase}
    return instantiate_from_config(data_config)
