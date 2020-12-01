#!/bin/bash
set -ex
pushd "$(cd $(dirname $0); pwd)/.." 1>/dev/null
mkdir -p $(arch)
source fj-misc/$(arch).multi.conf
source ${PYTORCH_INSTALL_PATH}/${VENV_NAME}/bin/activate

TRAIN_DATA_PATH=${TRAIN_DATA_PATH:="data/demo"}
BATCH_SIZE=${BATCH_SIZE:=1100}

python train.py \
       ${PROFILE_OPS} -data ${TRAIN_DATA_PATH} -save_model demo-model-transformer \
       -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
       -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 \
       -batch_type tokens -normalization tokens \
       -optim adam -adam_beta2 0.998 -decay_method noam \
       -warmup_steps 5 -learning_rate 2 \
       -max_grad_norm 0 -param_init 0 -param_init_glorot \
       -label_smoothing 0.1 --report_every 1 -batch_size ${BATCH_SIZE} -train_steps 10 \
       -cpu_backend mpi -world_size $WORLD_SIZE -cpu_rank $WORLD_RANK
