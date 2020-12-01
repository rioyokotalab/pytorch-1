#!/bin/bash

set -ex

# Install Rust
RUST_VERSION="1.43.1"
cd ${DOWNLOAD_PATH}/rust-${RUST_VERSION}-aarch64-unknown-linux-gnu
bash ./install.sh --prefix=${PREFIX}/.local
