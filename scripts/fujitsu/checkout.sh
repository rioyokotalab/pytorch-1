#!/bin/bash

# env
. env.src

# create common lib dir
mkdir -p ${PREFIX}/.local

# move download path
rm -fr ${DOWNLOAD_PATH}
mkdir ${DOWNLOAD_PATH}
cd ${DOWNLOAD_PATH}

# Download cmake
curl -O https://cmake.org/files/v3.11/cmake-3.11.4.tar.gz
tar zxf cmake-3.11.4.tar.gz

# Download python
curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
tar zxf Python-3.8.2.tgz

# Download jpegsrc
curl -O http://www.ijg.org/files/jpegsrc.v9d.tar.gz
tar zxf jpegsrc.v9d.tar.gz

# Download Pillow
git clone https://github.com/python-pillow/Pillow.git
cd Pillow
git checkout 7.2.0 -b 7.2.0
sed -i "s;JPEG_ROOT = None;JPEG_ROOT = \"${PREFIX}/.local/lib\";" setup.py
cd ../

# Download torchvision
git clone https://github.com/fujitsu/vision.git
cd vision
git checkout -b fujitsu_v0.8.1_for_a64fx origin/fujitsu_v0.8.1_for_a64fx
cd ../

# Download horovod
git clone -b fujitsu_v0.20.3_for_a64fx https://github.com/fujitsu/horovod.git
cd horovod
git submodule sync
git submodule update --init --recursive
cd ../

# Download oneDNN
git clone https://github.com/fujitsu/oneDNN.git
cd oneDNN
git checkout -b v2.1.0L01_aarch64 refs/tags/v2.1.0L01_aarch64
git submodule sync
git submodule update --init --recursive
cd ../

# merge oneDNN to pytorch
cd ${PYTORCH_INSTALL_PATH}
cd ../../
git submodule sync
git submodule update --init --recursive
cd third_party/ideep/
rm -rf mkl-dnn
cp -rf ${DOWNLOAD_PATH}/oneDNN mkl-dnn
