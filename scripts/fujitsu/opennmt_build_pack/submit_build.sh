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
bash 5_pyonmttok.sh     # Pyonmttok
bash 6_h5py.sh          # H5PY
bash 7_grpc.sh          # GRPC
bash 8_torchtext.sh     # TorchText
bash 9_check.sh         # Check
