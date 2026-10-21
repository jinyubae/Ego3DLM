from .base import BaseLosses, PassThroughLoss


class LMLosses(BaseLosses):
    """Loss of the LM stages (I-III): the language-model / GRPO loss returned by the model."""

    def __init__(self, cfg, num_joints, **kwargs):
        losses = ["gpt_loss"]
        params = {'gpt_loss': cfg.LOSS.LAMBDA_CLS}
        super().__init__(cfg, losses, params, {'gpt_loss': PassThroughLoss}, num_joints, **kwargs)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0
        total += self._update_loss("gpt_loss", rs_set['outputs'].loss, rs_set['outputs'].loss)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
