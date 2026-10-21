import torch
from ego3dlm.config import instantiate_from_config
from ego3dlm.losses.pqvae import PQVAELosses
from .base import BaseModel


class MotionTokenizer(BaseModel):
    """
    Stage 0: motion PQ-VAE tokenizer (also used to tokenize all motions, scripts/tokenize_motion.py)
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 metrics_dict=['MRMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=True)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        self.vae = instantiate_from_config(motion_vae)

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: PQVAELosses(cfg, self.hparams.stage, self.datamodule.njoints, self.datamodule.unpack_features)
            for split in ["losses_train", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        joints_ref = self.feats2joints(feats_ref)
        # motion encode & decode
        feats_rst, loss_commit, perplexity = self.vae(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        return {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
        }

    @torch.no_grad()
    def val_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion encode & decode, sample by sample at its own length
        feats_rst = torch.zeros_like(feats_ref)
        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

        return {
            "joints_ref": self.feats2joints(feats_ref),
            "joints_rst": self.feats2joints(feats_rst),
            "length": lengths,
        }

    def allsplit_step(self, split: str, batch, batch_idx):
        # Reconstruction losses (also logged for the validation split)
        rs_set = self.train_vae_forward(batch)
        loss = self._losses['losses_' + split].update(rs_set)

        # Reconstruction metrics
        if split == "val":
            rs_set = self.val_vae_forward(batch)
            for metric in self.hparams.metrics_dict:
                getattr(self.metrics, metric).update(rs_set["joints_rst"], rs_set["joints_ref"], batch['length'])

        return loss
