import os
import pickle
import types
import warnings
from collections import OrderedDict

import torch
from pytorch_lightning.plugins.io import TorchCheckpointIO


class _UnavailableObject:
    """Stand-in for a pickled object whose class cannot be imported any more, such as the word vectorizer kept in
    the data-module hyper-parameters of checkpoints written by an older layout of this code. Only the weights of a
    checkpoint are used, so these objects are not needed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _checkpoint_pickle_module():
    """pickle module for torch.load that replaces classes it cannot import with _UnavailableObject."""

    class Unpickler(pickle.Unpickler):  # pickle.Unpickler as patched by Lightning, which redirects its old names
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                warnings.warn(f"{module}.{name} in the checkpoint cannot be imported; it is skipped "
                              "(only the weights are used)")
                return _UnavailableObject

    module = types.ModuleType("checkpoint_pickle")
    module.Unpickler = Unpickler
    module.load = lambda file, **kwargs: Unpickler(file, **kwargs).load()
    return module


def load_checkpoint_file(path, map_location="cpu"):
    """torch.load of a checkpoint, also of one written by an older version of the code."""
    return torch.load(path, map_location=map_location, pickle_module=_checkpoint_pickle_module())


class CheckpointIO(TorchCheckpointIO):
    """Lightning checkpoint IO that reads checkpoints with load_checkpoint_file (resume, validation from ckpt_path)."""

    def load_checkpoint(self, path, map_location=lambda storage, loc: storage):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found.")
        return load_checkpoint_file(path, map_location=map_location)


def add_special_tokens(cfg, model):
    """Special tokens of Stages II / III: three-point, narration and scene-reasoning delimiters."""
    params = cfg.model.params
    if params.get("tp_special_token", False):
        model.lm.init_tp_special_token()
    if params.get("text_special_token", False):
        model.lm.init_text_special_token()
    if params.get("cot_special_token", False):
        model.lm.init_cot_special_token()


def load_pretrained(cfg, model, logger=None):
    """Load the previous-stage checkpoint (non-strict; see the key filters below)."""
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")

    state_dict = load_checkpoint_file(cfg.TRAIN.PRETRAINED)["state_dict"]

    ignore_prefixes = []

    # Stage II from Stage I: the three-point encoder input changes (27 -> 54), so it is re-initialized
    if cfg.TRAIN.get('LOAD_CKPT_WO_TP_ENC', False):
        ignore_prefixes.append("three_points_proj.")

    # GRPO validation: exclude ref_model weights from checkpoint
    if cfg.TRAIN.get('GRPO_VAL', False):
        ignore_prefixes.append("ref_model.")

    if ignore_prefixes:
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(k.startswith(p) for p in ignore_prefixes)}
        if logger is not None:
            logger.info(f"Filtered out keys with prefixes: {ignore_prefixes}")
    model.load_state_dict(state_dict, strict=False)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    state_dict = load_checkpoint_file(cfg.TRAIN.PRETRAINED_VAE)['state_dict']
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
    model.vae.load_state_dict(vae_dict, strict=True)

    return model


def load_pretrained_grpo_ref(cfg, model, logger=None):
    """Load GRPO reference model from checkpoint."""
    ref_path = cfg.TRAIN.GRPO.REF_MODEL_PATH
    ref_device = cfg.TRAIN.GRPO.get("REF_DEVICE", False)
    
    if not ref_device :
        ref_device = model.device
    if logger is not None:
        logger.info(f"Loading GRPO reference model from {ref_path}")
        logger.info(f"Reference model will be placed on {ref_device}")

    from ego3dlm.models.build_model import build_model

    # Create ref model with same architecture as training model
    ref_model = build_model(cfg, model.datamodule)

    # Add same special tokens as training model
    add_special_tokens(cfg, ref_model)

    # Load state dict (vocab size now matches checkpoint)
    state_dict = load_checkpoint_file(ref_path)["state_dict"]
    ref_model.load_state_dict(state_dict, strict=True)

    ref_model = ref_model.to(ref_device)
    ref_model.eval()

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    model.ref_model = ref_model
    model.ref_device = ref_device

    return model
