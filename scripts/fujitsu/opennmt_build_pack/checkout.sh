#!/bin/bash

set -ex

# env
. env.src

# move download path
rm -fr ${DOWNLOAD_PATH}
mkdir -p ${DOWNLOAD_PATH}
cd ${DOWNLOAD_PATH}

# Download boost
BOOST_VERSION="1.74.0"
BOOST_TAR_VERSION="1_74_0"
wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_TAR_VERSION}.tar.gz
tar zxvf boost_${BOOST_TAR_VERSION}.tar.gz && rm boost_${BOOST_TAR_VERSION}.tar.gz

# Download rust
RUST_VERSION="1.43.1"
curl -O https://static.rust-lang.org/dist/rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.gz
tar zxvf rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.gz && rm rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.gz

# Download sentencepiece
SENTENCEPIECE_VERSION="v0.1.90"
git clone https://github.com/google/sentencepiece.git --depth 1 -b ${SENTENCEPIECE_VERSION}

# Download pyonmttok
PYONMTTOK_VERSION="v1.18.3"
git clone https://github.com/OpenNMT/Tokenizer.git --depth 1 -b ${PYONMTTOK_VERSION}

# Download hdf5
HDF5_MAJOR_VERSION="1.10"
HDF5_MINOR_VERSION="1.10.1"
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR_VERSION}/hdf5-${HDF5_MINOR_VERSION}/src/CMake-hdf5-${HDF5_MINOR_VERSION}.tar.gz
tar zxvf CMake-hdf5-${HDF5_MINOR_VERSION}.tar.gz && rm CMake-hdf5-${HDF5_MINOR_VERSION}.tar.gz

# Download h5py
H5PY_VERSION="2.8.0"
git clone https://github.com/h5py/h5py.git --depth 1 -b ${H5PY_VERSION}

# Download grpcio
GRPCIO_VERSION="v1.28.0"
git clone --recursive https://github.com/grpc/grpc.git --depth 1 -b ${GRPCIO_VERSION}

# Download torchtext
TORCHTEXT_VERSION="0.4.0"
git clone --recursive https://github.com/pytorch/text.git --depth 1 -b ${TORCHTEXT_VERSION}

# Download OpenNMT
OPENNMT_PY_VERSION="1.1.1"
git clone https://github.com/OpenNMT/OpenNMT-py.git --depth 1 -b ${OPENNMT_PY_VERSION}
