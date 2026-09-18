#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS="${WHEELS_DIR:-$REPO_ROOT/wheels}"


export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.0;7.5}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "CUDA_HOME           = $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST= $TORCH_CUDA_ARCH_LIST"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

# --- Build ---
mkdir -p "$WHEELS"
pip wheel --no-build-isolation --no-deps -w "$WHEELS" \
    "$REPO_ROOT/submodules/simple-knn" \
    "$REPO_ROOT/submodules/diff-gaussian-rasterization"

# --- Cai va kiem tra ---
pip install --no-deps --force-reinstall "$WHEELS"/*.whl
python - <<'PY'
import torch, simple_knn, diff_gaussian_rasterization
print("OK |", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

echo
echo "Wheel path: $WHEELS"
ls -lh "$WHEELS"