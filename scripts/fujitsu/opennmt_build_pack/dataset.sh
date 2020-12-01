#!/bin/bash

set -ex

# env
. env.src

# Download dataset
mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
wget https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz
tar zxvf wmt_ende_sp.tar.gz && rm wmt_ende_sp.tar.gz

# Shuffle
shuf --random-source=train.en train.en > train.en.shuf
shuf --random-source=train.en train.de > train.de.shuf
