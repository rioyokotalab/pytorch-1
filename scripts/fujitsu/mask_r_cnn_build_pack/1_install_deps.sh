#!/bin/bash

set -ex

# Install dependency
cd ${MASK_RCNN_PATH}

pip3 install --no-index --no-deps \
     --find-links=${UPLOAD_PATH} \
     -r ${UPLOAD_PATH}/requirements.txt
