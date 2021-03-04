#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=02:00:00
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
bash 2_freetype.sh      # freetype
bash 3_matplotlib.sh    # matplotlib
bash 4_pycocotools.sh   # pycocotools
bash 5_detectron2.sh    # detectron2
bash 6_check.sh         # Check
