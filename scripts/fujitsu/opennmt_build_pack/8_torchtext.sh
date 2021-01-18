#!/bin/bash

set -ex

# Install Torchtext
cd ${DOWNLOAD_PATH}/text
python3 setup.py clean
python3 setup.py install
