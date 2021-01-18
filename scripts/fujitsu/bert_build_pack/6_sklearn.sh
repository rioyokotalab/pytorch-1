#!/bin/bash

set -ex

# Numpy
# export NPY_NUM_BUILD_JOBS=48
# pip3 uninstall -y numpy

# cd ${DOWNLOAD_PATH}/numpy
# patch -p1 < ${UPLOAD_PATH}/numpy_ssl2_v2.patch
# python3 setup.py build --compiler=fj --fcompiler=fujitsu
# python3 setup.py install
# cd ..

# Install scipy
pip3 install ${UPLOAD_PATH}/scipy-1.5.2-cp38-cp38-linux_aarch64.whl
# TODO : avoid build error
# cd ${DOWNLOAD_PATH}/scipy
# python setup.py build
# python3 setup.py install
# cd ..
# python3 -c 'import scipy'

# sklearn
pip3 install ${UPLOAD_PATH}/scikit_learn-0.22.1-cp38-cp38-linux_aarch64.whl
# TODO : avoid build error
# cd ${DOWNLOAD_PATH}/scikit-learn
# pip3 install .
# cd ..
# python3 -c 'import sklearn'
