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
pushd third_party/ideep/mkl-dnn/src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party
mkdir -p build_xed_aarch64
cd build_xed_aarch64/
../xed/mfile.py --shared  --cc="${TCSDS_PATH}/bin/fcc -Nclang -Kfast -Knolargepage" --cxx="${TCSDS_PATH}/bin/FCC -Nclang -Kfast -Knolargepage" examples install
cd kits
ln -sf xed-install-base-* xed
cd ../../../../
make -j48

popd
export XED_ROOT_DIR=`pwd`/third_party/ideep/mkl-dnn/src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/build_xed_aarch64/kits/xed/lib
ln -sf ${XED_ROOT_DIR}/libxed.so ${PREFIX}/.local/lib/libxed.so

python3 setup.py clean
python3 setup.py install

