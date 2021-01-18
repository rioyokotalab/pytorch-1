#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L "elapse=02:00:00"
#PJM -L "node=1:noncont"
#PJM -j
#PJM -S

set -ex

# Resource Size
ulimit -s 8192

# Library
source fj-misc/$(arch).multi.conf

# Virtual env
source ${PYTORCH_INSTALL_PATH}/${VENV_NAME}/bin/activate

#######################
# Pre-Process
#######################
python3 preprocess.py \
	-train_src ${DATA_DIR}/train.en.shuf \
	-train_tgt ${DATA_DIR}/train.de.shuf \
	-valid_src ${DATA_DIR}/valid.en \
	-valid_tgt ${DATA_DIR}/valid.de \
	-save_data ${DATA_DIR}/processed_shard \
	-src_seq_length 100 \
	-tgt_seq_length 100 \
	-shard_size 500000 \
	-share_vocab \
	-overwrite
