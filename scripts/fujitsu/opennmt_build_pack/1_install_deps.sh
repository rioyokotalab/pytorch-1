#!/bin/bash

set -ex

# Install dependency
cd ${OPENNMT_PATH}
pip3 install --no-index --no-deps \
     --find-links=${UPLOAD_PATH} \
     -r ${UPLOAD_PATH}/requirements.txt
