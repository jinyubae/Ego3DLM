#!/bin/bash
# GloVe word vectors (our_vab_*) used by the text-motion evaluator, taken from the T2M evaluator archive
# distributed by MotionGPT. Creates deps/glove.
set -e
mkdir -p deps/
cd deps/
if [[ -f glove/our_vab_data.npy ]]; then echo "deps/glove exists"; exit 0; fi
echo "Downloading the GloVe word vectors into ./deps/glove"
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
tar xfz t2m.tar.gz
mv t2m/glove glove
rm -rf t2m t2m.tar.gz
echo "Done."
