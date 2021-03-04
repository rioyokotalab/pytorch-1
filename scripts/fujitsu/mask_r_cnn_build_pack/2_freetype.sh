#!/bin/bash

set -ex

# Install freetype
cd ${DOWNLOAD_PATH}/freetype-2.6.1

./configure --prefix=${PREFIX}/.local --enable-freetype-config
make -j48 && make install
