
# Build jpeg-9d
cd ${DOWNLOAD_PATH}
cd jpeg-9d/
./configure --prefix="${PREFIX}/.local" --enable-shared
make clean
make -j32
make install
