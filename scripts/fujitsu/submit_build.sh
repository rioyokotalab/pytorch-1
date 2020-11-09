#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=04:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

# env
. env.src

sh 1_cmake.sh  # cmake

export CC="${TCSDS_PATH}/bin/fcc -Nclang -Kfast"
export CXX="${TCSDS_PATH}/bin/FCC -Nclang -Kfast"
#export CC="${TCSDS_PATH}/bin/fcc -Nclang -Kfast -Knolargepage -lpthread"
#export CXX="${TCSDS_PATH}/bin/FCC -Nclang -Kfast -Knolargepage -lpthread"

sh 2_python.sh  # python
sh 3_venv.sh  # venv
sh 4_pytorch.sh  # pytorch(v150)
sh 5_jpeg.sh  # jpeg
sh 6_vision.sh  # vision
sh 7_horovod.sh  # horovod
sh 8_check.sh  # check
cp .libtcmalloc.so ${PREFIX}/.local/lib/libtcmalloc.so
