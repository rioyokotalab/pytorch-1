#!/bin/bash

set -ex

# env
. env.src

# move download path
rm -fr ${DOWNLOAD_PATH}
mkdir -p ${DOWNLOAD_PATH}
cd ${DOWNLOAD_PATH}

# Download freetype
wget https://downloads.sourceforge.net/project/freetype/freetype2/2.6.1/freetype-2.6.1.tar.gz
tar zxvf freetype-2.6.1.tar.gz && rm freetype-2.6.1.tar.gz

# Download matplotlib
git clone https://github.com/matplotlib/matplotlib.git --depth 1 -b v3.3.4
cd matplotlib
sed 's/\#system_freetype = False/system_freetype = True/g' setup.cfg.template > setup.cfg
cd ..

# Download pycocotools
wget https://files.pythonhosted.org/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
tar zxvf pycocotools-2.0.2.tar.gz && rm pycocotools-2.0.2.tar.gz

# Download detectron2
cd ${MASK_RCNN_PATH}
git clone https://github.com/facebookresearch/detectron2.git --depth 1 -b v0.2.1
cd detectron2
patch -p1 < ${UPLOAD_PATH}/detectron2_021.patch
patch -p1 < ${UPLOAD_PATH}/detectron2_021_fx700.patch
