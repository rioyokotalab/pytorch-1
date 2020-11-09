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
git checkout 6.2.1 -b 6.2.1
sed -i "s;JPEG_ROOT = None;JPEG_ROOT = \"${PREFIX}/.local/lib\";" setup.py
cd ../

# Download torchvision
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.5.0 -b v0.5.0
cd ../

# Download horovod
git clone https://github.com/horovod/horovod.git
cd horovod
git checkout -b v0.19.0 refs/tags/v0.19.0
patch -p1 < ${UPLOAD_PATH}/fj_hvd.patch
git submodule sync
git submodule update --init --recursive
cd ../

# Download dnnl_aarch64
git clone https://github.com/fujitsu/dnnl_aarch64.git
cd dnnl_aarch64/
git checkout 549a467a5797900eadb21d1d1a0a6810fd3ac5be
git submodule sync
git submodule update --init --recursive
cd ../

# merge dnnl_aarch64 to pytorch
cd ${PYTORCH_INSTALL_PATH}
cd ../../
git submodule sync
git submodule update --init --recursive
cd third_party/ideep/
rm -rf mkl-dnn
cp -rf ${DOWNLOAD_PATH}/dnnl_aarch64 mkl-dnn
