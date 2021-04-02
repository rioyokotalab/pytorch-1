#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L "elapse=02:00:00"
#PJM -L "node=1:noncont"
#PJM --mpi "max-proc-per-node=2"
#PJM -j
#PJM -S

set -ex

# Resource Size
ulimit -s 8192

# Library
source fj-misc/$(arch).multi.conf

# Virtual env
source ${PYTORCH_INSTALL_PATH}/${VENV_NAME}/bin/activate

# Show Info
echo $(date) " ## Print Env"
env | grep -e ^PATH= -e ^LD_LIBRARY_PATH= -e ^LD_PRELOAD= | sed "s/:/\n  /g" | sed "s/=/\n  /g"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.__config__.show())"

#######################
# Run
#######################
mpirun -x BATCH_SIZE=3850 --map-by slot:PE=24 -np 2  ./fj-misc/run_transformer_multi_impl.sh
#mpirun -x BATCH_SIZE=3850 -np 4  ./fj-misc/run_transformer_multi_impl.sh
#mpirun -x BATCH_SIZE=3850 -np 8  ./fj-misc/run_transformer_multi_impl.sh
#mpirun -x BATCH_SIZE=3850 -np 16 ./fj-misc/run_transformer_multi_impl.sh
