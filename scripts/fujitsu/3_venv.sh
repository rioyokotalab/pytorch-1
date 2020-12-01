# Create venv
cd ${PYTORCH_INSTALL_PATH}
${PREFIX}/.local/bin/python3.8 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate

# Install requires
pip3 install ${UPLOAD_PATH}/PyYAML-5.3-cp38-cp38-linux_aarch64.whl
pip3 install ${UPLOAD_PATH}/numpy-1.18.4-cp38-cp38-linux_aarch64.whl
pip3 install ${UPLOAD_PATH}/cloudpickle-1.3.0-py2.py3-none-any.whl ${UPLOAD_PATH}/psutil-5.7.0-cp38-cp38-linux_aarch64.whl ${UPLOAD_PATH}/tqdm-4.46.0-py2.py3-none-any.whl ${UPLOAD_PATH}/cffi-1.14.0-cp38-cp38-linux_aarch64.whl ${UPLOAD_PATH}/pycparser-2.20-py2.py3-none-any.whl
pip3 install ${UPLOAD_PATH}/six-1.14.0-py2.py3-none-any.whl
