#!/bin/bash

set -ex

# Install matplotlib
cd ${DOWNLOAD_PATH}/matplotlib

export CFLAGS="-I${PREFIX}/.local/include/freetype2/ft2build.h"
export LDFLAGS="-L${PREFIX}/.local/lib/libfreetype.so"
pip install .
