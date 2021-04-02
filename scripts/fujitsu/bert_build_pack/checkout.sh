#!/bin/bash

set -ex

# env
source env.src

cd ${BERT_PATH}
git clone https://github.com/huggingface/transformers.git
cd transformers/
git checkout -b v3.4.0 refs/tags/v3.4.0
patch -p1  < ${UPLOAD_PATH}/transformers_v340.diff
patch -p1  < ${UPLOAD_PATH}/transformers_v340_fx700.diff

# move download path
rm -fr ${DOWNLOAD_PATH}
mkdir -p ${DOWNLOAD_PATH}
cd ${DOWNLOAD_PATH}

# Download boost
BOOST_VERSION="1.74.0"
BOOST_TAR_VERSION="1_74_0"
wget https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_TAR_VERSION}.tar.gz
tar zxvf boost_${BOOST_TAR_VERSION}.tar.gz && rm boost_${BOOST_TAR_VERSION}.tar.gz

# Download rust
RUST_VERSION="1.43.1"
curl -O https://static.rust-lang.org/dist/rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.gz
tar zxvf rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.gz && rm rust-${RUST_VERSION}-aarch64-unknown-linux-gnu.tar.gz

# Download sentencepiece
SENTENCEPIECE_VERSION="v0.1.90"
git clone https://github.com/google/sentencepiece.git --depth 1 -b ${SENTENCEPIECE_VERSION}

# Download Tokenizers
TOKENIZERS_VERSION="python-v0.9.2"
git clone https://github.com/huggingface/tokenizers.git --depth 1 -b ${TOKENIZERS_VERSION}

# Download numpy
SCIPY_VERSION="v1.19.0"
git clone https://github.com/numpy/numpy.git --depth 1 -b v1.19.0

# Download scipy
SCIPY_VERSION="v1.5.2"
git clone https://github.com/scipy/scipy.git --depth 1 -b ${SCIPY_VERSION}

# Download sklearn
SKLEARN_VERSION="0.22.1"
git clone https://github.com/scikit-learn/scikit-learn.git --depth 1 -b ${SKLEARN_VERSION}
