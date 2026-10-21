import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from ego3dlm.config import parse_args
from ego3dlm.data.build_data import build_data
from ego3dlm.models.build_model import build_model
from ego3dlm.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH)

    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    for batch in tqdm(datasets.train_dataloader(),
                      desc='motion tokenize'):
        name = batch['text']
        pose = batch['motion'].cuda().float()

        if pose.shape[1] == 0:
            continue
        target = model.vae.encode(pose)
        target = target.to('cpu').numpy()

        target_path = os.path.join(output_dir, name[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, target)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
