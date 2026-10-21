"""Strip a training checkpoint down to the weights needed for inference.

Removes the optimizer state, the frozen GRPO reference model (``ref_model.*``)
and the evaluator copies held by the metric modules (``metrics.*``; they are
re-loaded from ``METRIC.FEATURE_EXTRACTOR`` at run time).

Usage:
    python -m scripts.strip_checkpoint --src path/to/full.ckpt --dst path/to/release.ckpt
"""
import argparse

import torch

from ego3dlm.utils.load_checkpoint import load_checkpoint_file

DROP_PREFIXES = ("ref_model.", "metrics.")
KEEP_META = ("epoch", "global_step", "pytorch-lightning_version")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="full training checkpoint")
    parser.add_argument("--dst", required=True, help="output path of the stripped checkpoint")
    parser.add_argument("--no-verify", action="store_true", help="skip the reload check")
    args = parser.parse_args()

    ckpt = load_checkpoint_file(args.src)
    state_dict = ckpt["state_dict"]
    kept = {k: v for k, v in state_dict.items() if not k.startswith(DROP_PREFIXES)}

    out = {k: ckpt[k] for k in KEEP_META if k in ckpt}
    out["state_dict"] = kept
    torch.save(out, args.dst)

    n_drop = len(state_dict) - len(kept)
    size = sum(v.numel() * v.element_size() for v in kept.values()) / 1024**3
    print(f"kept {len(kept)} tensors ({size:.2f} GB), dropped {n_drop}")

    if not args.no_verify:
        reloaded = torch.load(args.dst, map_location="cpu")["state_dict"]
        assert reloaded.keys() == kept.keys()
        for k, v in kept.items():
            assert torch.equal(reloaded[k], v), k
        print("verified: all kept tensors are identical to the source checkpoint")


if __name__ == "__main__":
    main()
