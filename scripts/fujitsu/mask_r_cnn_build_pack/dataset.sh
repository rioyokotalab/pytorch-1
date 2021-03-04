#!/bin/bash

set -ex

# env
. env.src

# Download dataset
mkdir -p ${DATA_DIR}/coco && pushd ${DATA_DIR}/coco
curl -O http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
popd

# Download weights
mkdir -p ${WEIGHTS_DIR} && pushd ${WEIGHTS_DIR}
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
popd
