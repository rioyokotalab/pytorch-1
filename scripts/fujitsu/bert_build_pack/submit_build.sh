#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=04:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

# env
source env.src
source ${PYTORCH_INSTALL_PATH}/${VENV_NAME}/bin/activate

# make install directory
mkdir -p ${PREFIX}/.local
mkdir -p ${PREFIX}/.local/lib
mkdir -p ${PREFIX}/.local/bin
mkdir -p ${PREFIX}/.local/include
ln -sf ${PREFIX}/.local/lib ${PREFIX}/.local/lib64

bash 1_install_deps.sh  # Install Deps
bash 2_boost.sh         # Boost
bash 3_rust.sh          # Rust
bash 4_sentencepiece.sh # Sentencepiece
bash 5_tokenizers.sh    # Tokenizers
bash 6_sklearn.sh       # sklearn
bash 7_transformers.sh  # transformers
bash 8_check.sh         # Check
