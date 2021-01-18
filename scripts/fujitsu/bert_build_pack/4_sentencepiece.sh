#!/bin/bash

set -ex

# Install sentencepiece
cd ${DOWNLOAD_PATH}/sentencepiece
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${PREFIX}/.local

# make src
make clean
make -j48 && make install

# Install Python extention
export PKG_CONFIG_PATH=${DOWNLOAD_PATH}/sentencepiece/build
export CFLAGS="-L${PREFIX}/.local/lib -I${PREFIX}/.local/include"

cd ../python/
sed -i "s;cmd('pkg-config sentencepiece --cflags');[\"${CFLAGS}\"];g" setup.py
sed -i "s;cmd('pkg-config sentencepiece --libs');[\"-L/${INSTALL_DIR}/lib\", \"-lsentencepiece\", \"-lsentencepiece_train\"];g" setup.py

python3 setup.py install
