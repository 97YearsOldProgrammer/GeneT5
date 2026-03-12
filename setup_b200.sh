#!/bin/bash
# Setup script for RunPod B200 (Blackwell, sm_100, 192GB HBM3e)
#
# Container: nvcr.io/nvidia/pytorch:26.02-py3 (CUDA 13.1, PyTorch 2.11, Python 3.12)
#
# Attention cascade: FA4 > FA2 > SDPA  (FA3 is BLOCKED on Blackwell sm_100)
# MoE cascade:       triton > grouped_mm > bmm
#
# Usage: bash setup_b200.sh [/path/to/network/volume]

set -euo pipefail

NETWORK_VOL="${1:-/workspace}"
CODE_DIR="${NETWORK_VOL}/GeneT5"
PIP="pip install --break-system-packages -q"

echo "=== RunPod B200 Setup ==="
echo "Network volume: ${NETWORK_VOL}"

# ── System info ──
echo ""
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, SM: {torch.cuda.get_device_capability()}')"

SM_VERSION=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])")
if [ "$SM_VERSION" -lt 10 ]; then
    echo "ERROR: This script is for Blackwell (sm_100+). Detected sm_${SM_VERSION}x"
    echo "       Use setup_h100.sh for Hopper (sm_90)"
    exit 1
fi

# ── Step 1: Flash Attention 4 (prebuilt pip wheel) ──
echo ""
echo "--- Step 1: Flash Attention 4 ---"
if python3 -c "from flash_attn.cute.interface import flash_attn_func" 2>/dev/null; then
    echo "FA4 already installed"
else
    # FA4 conflicts with FA2 (flash-attn) namespace — uninstall FA2 first
    pip uninstall -y flash-attn 2>/dev/null || true

    echo "Installing FA4 (prebuilt wheel, no compilation)..."
    ${PIP} flash-attn-4==4.0.0b4 || {
        echo "Prebuilt FA4 failed, trying from source..."
        FA4_DIR="${NETWORK_VOL}/.cache/flash-attention"
        [ -d "${FA4_DIR}" ] || git clone https://github.com/Dao-AILab/flash-attention "${FA4_DIR}"
        ${PIP} -e "${FA4_DIR}/flash_attn/cute[dev]"
    }
fi

# ── Step 2: FA2 fallback (only if FA4 failed) ──
echo ""
echo "--- Step 2: FA2 fallback check ---"
if python3 -c "from flash_attn.cute.interface import flash_attn_func" 2>/dev/null; then
    echo "FA4 working — skipping FA2 install (they conflict)"
else
    echo "FA4 not available, installing FA2 as fallback..."
    ${PIP} flash-attn --no-build-isolation || echo "FA2 install failed — will use SDPA"
fi

# ── Step 3: Project dependencies ──
echo ""
echo "--- Step 3: Project dependencies ---"
${PIP} liger-kernel webdataset pyfaidx einops psutil

# ── Step 4: tmux (for persistent training sessions) ──
echo ""
echo "--- Step 4: tmux ---"
if command -v tmux &>/dev/null; then
    echo "tmux already installed"
else
    echo "Installing tmux..."
    apt-get update -qq && apt-get install -y -qq tmux
fi

# ── Step 5: Environment variables ──
echo ""
echo "--- Step 5: Environment ---"
BASHRC="${HOME}/.bashrc"

add_env() {
    local key="$1" val="$2"
    if ! grep -q "^export ${key}=" "$BASHRC" 2>/dev/null; then
        echo "export ${key}=${val}" >> "$BASHRC"
    fi
}

if [ -d "${CODE_DIR}" ]; then
    add_env "PYTHONPATH" "${CODE_DIR}:\$PYTHONPATH"
fi

# B200 optimal backends
add_env "GENET5_ATTN_BACKEND" "auto"
add_env "GENET5_MOE_BACKEND"  "auto"

# PyTorch memory
add_env "PYTORCH_CUDA_ALLOC_CONF" "expandable_segments:True,garbage_collection_threshold:0.8"
add_env "CUDA_MODULE_LOADING" "LAZY"

# Compile cache (persist on network volume)
add_env "TRITON_CACHE_DIR"         "${NETWORK_VOL}/.cache/triton"
add_env "TORCHINDUCTOR_CACHE_DIR"  "${NETWORK_VOL}/.cache/torchinductor"
add_env "TORCHINDUCTOR_FX_GRAPH_CACHE" "1"

mkdir -p "${NETWORK_VOL}/.cache/triton" "${NETWORK_VOL}/.cache/torchinductor"

# ── Step 6: Verification ──
echo ""
echo "--- Step 6: Verification ---"
python3 << 'PYEOF'
import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
cap = torch.cuda.get_device_capability()
print(f"GPU: {torch.cuda.get_device_name(0)}, SM {cap[0]}{cap[1]}, {torch.cuda.get_device_properties(0).total_mem // (1024**3)}GB")
print()

checks = []

# FA4 check
try:
    from flash_attn.cute.interface import flash_attn_func
    checks.append(("FA4", "OK"))
except Exception as e:
    checks.append(("FA4", f"FAIL - {e}"))

# FA2 check (may conflict with FA4)
try:
    from flash_attn import flash_attn_func as fa2
    checks.append(("FA2", "OK (installed alongside FA4 — may conflict)"))
except ImportError:
    checks.append(("FA2", "not installed (FA4 takes priority)"))
except Exception as e:
    checks.append(("FA2", f"FAIL - {e}"))

# FA3 — should NOT work on Blackwell
checks.append(("FA3", "BLOCKED on Blackwell sm_100 (expected)"))

# grouped_mm (always available on Hopper/Blackwell)
try:
    assert hasattr(torch, '_grouped_mm')
    checks.append(("grouped_mm", "OK"))
except Exception as e:
    checks.append(("grouped_mm", f"FAIL - {e}"))

# Liger
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    checks.append(("Liger", "OK"))
except Exception as e:
    checks.append(("Liger", f"FAIL - {e}"))

for name, status in checks:
    print(f"  {name}: {status}")

# FA4 smoke test
print()
try:
    from flash_attn.cute.interface import flash_attn_func as fa4
    q = torch.randn(2, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
    out = fa4(q, k, v, causal=False)
    if isinstance(out, tuple): out = out[0]
    print(f"FA4 smoke test: PASS ({out.shape})")
except Exception as e:
    print(f"FA4 smoke test: FAIL - {e}")
    # Try FA2 fallback
    try:
        from flash_attn import flash_attn_func as fa2
        q = torch.randn(2, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, 128, 12, 64, device="cuda", dtype=torch.bfloat16)
        out = fa2(q, k, v, causal=False)
        if isinstance(out, tuple): out = out[0]
        print(f"FA2 smoke test: PASS ({out.shape})")
    except Exception as e2:
        print(f"FA2 smoke test: FAIL - {e2}")
        print("Will use SDPA (torch built-in)")

# grouped_mm smoke test
try:
    x  = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16)
    w  = torch.randn(8, 768, 1536, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64], dtype=torch.int32, device="cuda")
    out = torch._grouped_mm(x, w, offs=offs)
    print(f"grouped_mm smoke test: PASS ({out.shape})")
except Exception as e:
    print(f"grouped_mm smoke test: FAIL - {e}")

PYEOF

# ── Summary ──
echo ""
echo "=== Setup complete ==="
echo ""
echo "B200 optimal config:"
echo "  Attention: FA4 (auto-detected) > FA2 > SDPA"
echo "  MoE:       triton > grouped_mm > bmm"
echo ""
echo "  GENET5_ATTN_BACKEND=auto"
echo "  GENET5_MOE_BACKEND=auto"
echo ""
echo "B200 vs H100 differences:"
echo "  - FA3 is BLOCKED on B200 (sm_100 excluded)"
echo "  - FA4 replaces FA3 (faster on Blackwell)"
echo "  - pin_memory=True (HBM, not UMA)"
echo "  - torch.compile dynamic=True OK (not ARM)"
echo "  - 192GB HBM3e (plenty for 371M model)"
echo ""
echo "Launch training in tmux (survives terminal disconnect):"
echo "  bash train/run_b200.sh <data_dir> <output_dir> <model_path> [flags]"
echo ""
echo "Source env: source ~/.bashrc"
