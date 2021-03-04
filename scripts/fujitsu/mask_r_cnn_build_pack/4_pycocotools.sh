#!/bin/bash

set -ex

# Install matplotlib
cd ${DOWNLOAD_PATH}/pycocotools-2.0.2

python setup.py build --parallel 48
python setup.py install
