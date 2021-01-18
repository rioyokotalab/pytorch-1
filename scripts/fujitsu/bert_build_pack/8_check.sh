#!/bin/bash

set -ex

export LD_PRELOAD=${PREFIX}/.local/lib/libtcmalloc_minimal.so

# Check Install
python3 -c 'import sentencepiece'
python3 -c 'import transformers'
echo "check done!"
