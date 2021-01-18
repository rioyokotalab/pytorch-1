#!/bin/bash

set -ex

# Install dependency
cd ${BERT_PATH}
pip3 install --no-index --no-deps \
     --find-links=${UPLOAD_PATH} \
     -r ${UPLOAD_PATH}/requirements.txt

pip3 install ${UPLOAD_PATH}/regex-2020.11.13-cp38-cp38-linux_aarch64.whl
