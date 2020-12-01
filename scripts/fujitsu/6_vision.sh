
# Load venv
cd ${PYTORCH_INSTALL_PATH}
cd ${VENV_NAME}
source bin/activate

# Install Pillow
cd ${DOWNLOAD_PATH}
cd Pillow
MAX_CONCURRENCY=8 CFLAGS="-I${PREFIX}/.local/include" python3 setup.py install
cd ../

# Install torchvision
cd vision
python3 setup.py clean
python3 setup.py install
