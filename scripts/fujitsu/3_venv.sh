# Create venv
cd ${PYTORCH_INSTALL_PATH}
${PREFIX}/.local/bin/python3.8 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate

# Install requires
pip3 install ${UPLOAD_PATH}/PyYAML-5.3.1-cp38-cp38-linux_aarch64.whl
pip3 install ${UPLOAD_PATH}/numpy-1.19.0-cp38-cp38-linux_aarch64.whl
pip3 install ${UPLOAD_PATH}/cloudpickle-1.3.0-py2.py3-none-any.whl ${UPLOAD_PATH}/psutil-5.7.0-cp38-cp38-linux_aarch64.whl ${UPLOAD_PATH}/tqdm-4.46.0-py2.py3-none-any.whl ${UPLOAD_PATH}/cffi-1.14.0-cp38-cp38-linux_aarch64.whl ${UPLOAD_PATH}/pycparser-2.20-py2.py3-none-any.whl
pip3 install ${UPLOAD_PATH}/six-1.14.0-py2.py3-none-any.whl
pip3 install ${UPLOAD_PATH}/Cython-0.29.21-py2.py3-none-any.whl ${UPLOAD_PATH}/attrs-20.3.0-py2.py3-none-any.whl ${UPLOAD_PATH}/dataclasses-0.6-py3-none-any.whl ${UPLOAD_PATH}/hypothesis-6.0.2-py3-none-any.whl ${UPLOAD_PATH}/sortedcontainers-2.3.0-py2.py3-none-any.whl ${UPLOAD_PATH}/typing_extensions-3.7.4.3-py3-none-any.whl
