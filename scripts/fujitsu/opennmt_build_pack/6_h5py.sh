#!/bin/bash

set -ex

# Install hdf5
HDF5_MAJOR_VERSION="1.10"
HDF5_MINOR_VERSION="1.10.1"
cd ${DOWNLOAD_PATH}/CMake-hdf5-${HDF5_MINOR_VERSION}
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX}/.local -DBUILD_SHARED_LIBS:BOOL=ON \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      -DHDF5_BUILD_TOOLS:BOOL=ON ../hdf5-${HDF5_MINOR_VERSION}

# make src
make clean
make -j32 && make install

# Install Python extention
export CFLAGS="-L${PREFIX}/.local/lib -I${PREFIX}/.local/include"

cd ${DOWNLOAD_PATH}/h5py
python3 setup.py configure --hdf5=${PREFIX}/.local
python3 setup.py configure --hdf5-version=${HDF5_MINOR_VERSION}
python3 setup.py install
