
# Build cmake
cd ${DOWNLOAD_PATH}/cmake-3.11.4
./configure --prefix=${PREFIX}/.local
make clean
make -j32
make install
