#!/bin/bash

set -ex

# Check Install
python3 -c 'import matplotlib'
python3 -c 'import matplotlib.dates'
python3 -c "import pycocotools"
python3 -c "import detectron2"
echo "check done!"
