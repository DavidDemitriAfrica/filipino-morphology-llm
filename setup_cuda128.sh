#!/bin/bash
# Setup script for CUDA 12.8 installation, cuDNN install, and module creation
# Runs idempotently and keeps the flow easy to follow.

set -euo pipefail

cd /scratch_aisg/SPEC-SF-AISG/railey

echo "[CUDA128] Starting CUDA 12.8 + cuDNN setup"

WORK_ROOT=$(pwd)
CUDA_DIR="$WORK_ROOT/cuda/cuda-toolkit-12.8"
MODULE_DIR="$WORK_ROOT/modules/cuda"
CUDNN_RPM="${CUDNN_RPM:-cudnn-local-repo-rhel10-9.16.0-1.0-1.x86_64.rpm}"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/9.16.0/local_installers/${CUDNN_RPM}"

echo ""
echo "[CUDA128] Step 1: Ensuring CUDA installer is available"
if [ ! -f "$WORK_ROOT/cuda_12.8.0_570.86.10_linux.run" ]; then
    echo "  - Downloading CUDA 12.8 installer"
    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run \
        -O "$WORK_ROOT/cuda_12.8.0_570.86.10_linux.run"
else
    echo "  - CUDA installer already present"
fi

echo ""
echo "[CUDA128] Step 2: Installing (or reusing) CUDA 12.8 toolkit"
mkdir -p "$CUDA_DIR"
NEED_CUDA_INSTALL=1
if [ -x "$CUDA_DIR/bin/nvcc" ] && "$CUDA_DIR/bin/nvcc" --version | grep -q "release 12.8"; then
    NEED_CUDA_INSTALL=0
    echo "  - Detected existing CUDA 12.8 installation at $CUDA_DIR"
fi
if [ "$NEED_CUDA_INSTALL" -eq 1 ]; then
    echo "  - Running installer"
    sh "$WORK_ROOT/cuda_12.8.0_570.86.10_linux.run" --silent --toolkit --installpath="$CUDA_DIR"
else
    echo "  - Skipping installer; version already matches"
fi
if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_DIR/bin/nvcc"
    exit 1
fi
nvcc --version

echo ""
echo "[CUDA128] Step 3: Ensuring cuDNN RPM is available"
if [ ! -f "$WORK_ROOT/$CUDNN_RPM" ]; then
    echo "  - Downloading $CUDNN_RPM"
    wget "$CUDNN_URL" -O "$WORK_ROOT/$CUDNN_RPM"
else
    echo "  - cuDNN RPM already present"
fi

echo ""
echo "[CUDA128] Step 4: Installing cuDNN payload"
if ! command -v rpm2cpio >/dev/null 2>&1; then
    echo "ERROR: rpm2cpio is required to extract the cuDNN RPM"
    exit 1
fi
tmpdir=$(mktemp -d)
(
    set -e
    cd "$tmpdir"
    rpm2cpio "$WORK_ROOT/$CUDNN_RPM" | cpio -idm >/dev/null 2>&1
)
CUDNN_TAR=$(find "$tmpdir" -name 'cudnn-linux-x86_64-*.tar.xz' -print -quit)
if [ -z "$CUDNN_TAR" ]; then
    echo "ERROR: Could not locate embedded cuDNN tarball in RPM"
    rm -rf "$tmpdir"
    exit 1
fi
tar -xJf "$CUDNN_TAR" -C "$tmpdir"
CUDNN_EXTRACT_DIR=$(find "$tmpdir" -maxdepth 1 -mindepth 1 -type d -name 'cudnn-linux-x86_64-*' -print -quit)
if [ -z "$CUDNN_EXTRACT_DIR" ]; then
    echo "ERROR: Could not extract cuDNN payload"
    rm -rf "$tmpdir"
    exit 1
fi
cp -P "$CUDNN_EXTRACT_DIR"/cuda/include/* "$CUDA_DIR"/include/
cp -P "$CUDNN_EXTRACT_DIR"/cuda/lib64/libcudnn* "$CUDA_DIR"/lib64/
chmod a+r "$CUDA_DIR"/include/cudnn*.h "$CUDA_DIR"/lib64/libcudnn*
rm -rf "$tmpdir"

echo ""
echo "[CUDA128] Step 5: Verifying cuDNN files"
if ! ls "$CUDA_DIR/lib64"/libcudnn.so* >/dev/null 2>&1; then
    echo "ERROR: libcudnn shared library not found in $CUDA_DIR/lib64"
    exit 1
fi
if [ ! -f "$CUDA_DIR/include/cudnn_version.h" ]; then
    echo "ERROR: cudnn_version.h not found in $CUDA_DIR/include"
    exit 1
fi
CUDNN_VERSION=$(CUDNN_HEADER="$CUDA_DIR/include/cudnn_version.h" python - <<'PY'
import os
import re
from pathlib import Path
header = Path(os.environ["CUDNN_HEADER"])
text = header.read_text()
major = re.search(r"#define CUDNN_MAJOR\s+(\d+)", text)
minor = re.search(r"#define CUDNN_MINOR\s+(\d+)", text)
patch = re.search(r"#define CUDNN_PATCHLEVEL\s+(\d+)", text)
if major and minor and patch:
    print(f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}")
else:
    print("unknown")
PY
)
if [ "$CUDNN_VERSION" = "unknown" ]; then
    echo "WARNING: Unable to parse cuDNN version from cudnn_version.h"
else
    echo "  - Detected cuDNN version $CUDNN_VERSION"
fi

echo ""
echo "[CUDA128] Step 6: Refreshing module files"
mkdir -p "$MODULE_DIR"
cat <<'EOF' > "$MODULE_DIR/12.8"
#%Module1.0
proc ModulesHelp { } {
    puts stderr "Sets up CUDA 12.8 locally"
}
module-whatis "CUDA Toolkit 12.8 with cuDNN"

set root /scratch_aisg/SPEC-SF-AISG/railey/cuda/cuda-toolkit-12.8

prepend-path PATH            $root/bin
prepend-path LD_LIBRARY_PATH $root/lib64
prepend-path LD_LIBRARY_PATH $root/lib
prepend-path LIBRARY_PATH    $root/lib64
prepend-path CPATH           $root/include
setenv       CUDA_HOME       $root
setenv       CUDA_ROOT       $root
setenv       CUDA_PATH       $root
EOF

export MODULEPATH="$WORK_ROOT/modules:$MODULEPATH"
module avail cuda || true
module load cuda/12.8
nvcc --version

CUDNN_SO=$(ls "$CUDA_DIR/lib64"/libcudnn.so* | head -n 1)
echo "[CUDA128] Checking cuDNN shared library linkage"
set +e
ldd "$CUDNN_SO"
status=$?
set -e
if [ $status -ne 0 ]; then
    echo "WARNING: ldd returned non-zero status when inspecting $CUDNN_SO"
fi

echo ""
echo "[CUDA128] Setup complete"