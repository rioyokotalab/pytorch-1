export USE_LAPACK=1
export USE_NNPACK=0
export USE_XNNPACK=0
export USE_NATIVE_ARCH=1
export MAX_JOBS=48

# Create venv
cd ${PYTORCH_INSTALL_PATH}
cd ${VENV_NAME}
source bin/activate

# Build PyTorch
cd ${PYTORCH_INSTALL_PATH}
cd ../../
cd third_party/ideep/
cd mkl-dnn/third_party/
mkdir -p build_xed_aarch64
cd build_xed_aarch64/
../xbyak_translator_aarch64/translator/third_party/xed/mfile.py --shared examples install --cc="${TCSDS_PATH}/bin/fcc -Nclang -Kfast -Knolargepage" --cxx="${TCSDS_PATH}/bin/FCC -Nclang -Kfast -Knolargepage"
cd kits/
ln -sf xed-install-base-* xed
cd ../../../../../../
python3 setup.py clean
python3 setup.py install
ln -sf ${PYTORCH_INSTALL_PATH}/../../third_party/ideep/mkl-dnn/third_party/build_xed_aarch64/kits/xed/lib/libxed.so ${PREFIX}/.local/lib/libxed.so
