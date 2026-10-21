#!/bin/bash
# Download the released Ego3DLM checkpoints and annotations from the Hugging Face Hub
# (https://huggingface.co/datasets/jinyubae/Ego3DLM) and place them where the configs expect them.
#
#   checkpoints/ego3dlm/stage3_grpo.ckpt                                   final Ego3DLM model (Stage III)
#   checkpoints/ego3dlm/stage0_pqvae.ckpt                                  motion PQ-VAE tokenizer (Stage 0)
#   checkpoints/evaluator/text_mot_match_trainset_only_len20/finest.tar    text-motion evaluator
#   datasets/ego3dlm/{splits, sample-id mapping, narrations, stats, templates, video_texts/*/qa_dataset.final.filtered.json}
#   obstacle_labels/*.txt                                                  spatial (obstacle / free-space) labels
#
# Usage: bash scripts/download_release.sh [checkpoints|annotations|all]   (uses huggingface_hub, installed with transformers)
set -e
WHAT=${1:-all}
REPO=jinyubae/Ego3DLM

fetch() {  # fetch <path in the HF repo> <output path>
  if [[ -f "$2" ]]; then echo "[skip] $2 exists"; return; fi
  mkdir -p "$(dirname "$2")"
  python - "$REPO" "$1" "$2" <<'EOF'
import os, shutil, sys, tempfile
from huggingface_hub import hf_hub_download
repo, filename, out = sys.argv[1:]
with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
    shutil.move(hf_hub_download(repo, filename, repo_type="dataset", local_dir=tmp), out)
EOF
  echo "$2"
}

if [[ "$WHAT" == "checkpoints" || "$WHAT" == "all" ]]; then
  fetch stage3_grpo/stage3_grpo.ckpt checkpoints/ego3dlm/stage3_grpo.ckpt
  fetch stage0_pqvae/stage0_pqvae.ckpt checkpoints/ego3dlm/stage0_pqvae.ckpt
  fetch evaluator/text_mot_match_trainset_only_len20/finest.tar checkpoints/evaluator/text_mot_match_trainset_only_len20/finest.tar
fi

if [[ "$WHAT" == "annotations" || "$WHAT" == "all" ]]; then
  tmp=$(mktemp -d)
  fetch annotations/ego3dlm_annotations.tar.gz "$tmp/ego3dlm_annotations.tar.gz"
  tar -xzf "$tmp/ego3dlm_annotations.tar.gz" -C "$tmp"
  mkdir -p datasets/ego3dlm obstacle_labels
  cp -r "$tmp"/ego3dlm_annotations/ego3dlm/. datasets/ego3dlm/
  cp -r "$tmp"/ego3dlm_annotations/obstacle_labels/. obstacle_labels/
  rm -rf "$tmp"
fi
echo "Done."
