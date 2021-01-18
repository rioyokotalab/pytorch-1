#!/bin/bash

set -ex

# Install pyonmttok
cd ${DOWNLOAD_PATH}/Tokenizer
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release .. \
      -DCMAKE_INSTALL_PREFIX=${PREFIX}/.local

# make src
make clean
make -j32 && make install

# Install Python extention
export CFLAGS="-L${PREFIX}/.local/lib -I${PREFIX}/.local/include"
cd ../bindings/python/
python3 setup.py install
