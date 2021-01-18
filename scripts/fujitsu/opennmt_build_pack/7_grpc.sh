#!/bin/bash

set -ex

# Install GRPC
cd ${DOWNLOAD_PATH}/grpc
sed -i 's/-std=gnu99//g' setup.py
python3 setup.py clean
export GRPC_PYTHON_BUILD_WITH_CYTHON=1
python3 setup.py build --parallel 32
python3 setup.py install
