#!/bin/bash

set -ex

# Install tokenizers
pip3 install ${UPLOAD_PATH}/tokenizers-0.9.2-cp38-cp38-linux_aarch64.whl

# TODO : Rust use network
# cd ${DOWNLOAD_PATH}/tokenizers/bindings/python
# python3 setup.py build --parallel 48
# python3 setup.py install
