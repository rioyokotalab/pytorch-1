#!/bin/bash

set -ex

# Install Boost
BOOST_VERSION="1.74.0"
BOOST_TAR_VERSION="1_74_0"

cd ${DOWNLOAD_PATH}/boost_${BOOST_TAR_VERSION}

./bootstrap.sh --prefix=${PREFIX}/.local --without-libraries=python
./b2 --clean
./b2 install -j32
