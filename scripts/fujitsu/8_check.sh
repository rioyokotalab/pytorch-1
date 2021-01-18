
# Load venv
cd ${PYTORCH_INSTALL_PATH}
cd ${VENV_NAME}
source bin/activate

# check
python3 -c 'import torch; print("PyTorch: ",torch.__version__)'
python3 -c 'import torchvision'
python3 -c 'import horovod.torch as hvd'
echo "check done!"
