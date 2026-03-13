#!/bin/bash
#
# One-click: reinit model (with <dmask> token) + distributed RAM bake
#
# Usage (on DGX Spark master, inside gt5 container):
#   bash train/init_and_bake.sh
#

set -euo pipefail

# ── Config ──
WORKER_IP="192.168.100.11"
RAW_DIR="/workspace/raw"
MODEL_DIR="model/init_dense_24L"
BAKE_OUTPUT="/workspace/baked/GeneT5/mar11_s51_w20k_ram"
VAL_SPECIES="B.taurus"
WINDOW_SIZE=20000

# Auto-detect code directory
CODE_DIR=""
for d in /workspace/Code/GeneT5 /workspace/GeneT5; do
    [ -f "$d/train/bake_data" ] && CODE_DIR="$d" && break
done
if [ -z "$CODE_DIR" ]; then
    CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR"

echo "============================================================"
echo "   GeneT5 Init + RAM Bake"
echo "============================================================"
echo "  Code:        $CODE_DIR"
echo "  Raw data:    $RAW_DIR"
echo "  Model init:  $MODEL_DIR"
echo "  Bake output: $BAKE_OUTPUT"
echo "  Val species: $VAL_SPECIES"
echo "  Window:      $WINDOW_SIZE bp"
echo "============================================================"
echo ""

# ── Step 1: Reinit model (DNABERT-2 4096 BPE + 11 special = 4107) ──
echo "============================================================"
echo "  Step 1: Reinitializing model (vocab includes <dmask>)"
echo "============================================================"

python init/init_model.py \
    --save_dir "$MODEL_DIR" \
    --layers 24 \
    --depth_scaling

echo ""

# ── Step 2: Distributed RAM bake ──
echo "============================================================"
echo "  Step 2: Distributed RAM bake (2 nodes)"
echo "============================================================"

bash train/bake.sh \
    --worker "$WORKER_IP" \
    "$RAW_DIR" \
    --tokenizer "$MODEL_DIR" \
    --output_dir "$BAKE_OUTPUT" \
    --window_size "$WINDOW_SIZE" \
    --ram \
    --val_species "$VAL_SPECIES"

# ── Summary ──
echo ""
echo "============================================================"
echo "  All Done"
echo "============================================================"
echo ""
echo "  Model:  $MODEL_DIR/"
echo "  Data:   $BAKE_OUTPUT/"
echo ""
echo "  Upload to H100:"
echo "    rsync -Pz $BAKE_OUTPUT/{tokens.bin,offsets.npy,prefix_lens.npy,metadata.json,validation.bin,eval.json,bake_config.json} runpod:/workspace/baked/GeneT5/mar11_s51_w20k_ram/"
echo "============================================================"
