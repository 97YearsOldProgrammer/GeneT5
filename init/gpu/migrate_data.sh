#!/bin/bash
#
# Migrate from flat /home/cg666/Data/ layout to per-project directories
#
# Run on BOTH DGX Sparks:
#   bash /home/cg666/Code/GeneT5/init/gpu/migrate_data.sh
#
# Idempotent: safe to run multiple times

set -euo pipefail

DATA="/home/cg666/Data"

echo "=== DGX Data Migration ==="
echo "Host: $(hostname)"
echo ""

# Create per-project directories
echo "[1/4] Creating per-project directories..."
mkdir -p "$DATA"/{GeneT5,FlashAlignment,ProteinQC}/{baked,model,logs}
mkdir -p "$DATA"/FlashAlignment/data
mkdir -p "$DATA"/ProteinQC/data

# GeneT5: move data into project dir
echo "[2/4] Moving GeneT5 data..."

if [ -d "$DATA/genome/baked/GeneT5" ] && [ ! -L "$DATA/genome/baked/GeneT5" ]; then
    echo "  Moving baked data..."
    mv "$DATA"/genome/baked/GeneT5/* "$DATA"/GeneT5/baked/ 2>/dev/null || true
    rmdir "$DATA/genome/baked/GeneT5" 2>/dev/null || true
else
    echo "  Baked data already migrated (or symlink)"
fi

if [ -d "$DATA/model/GeneT5" ] && [ ! -L "$DATA/model/GeneT5" ]; then
    echo "  Moving model data..."
    mv "$DATA"/model/GeneT5/* "$DATA"/GeneT5/model/ 2>/dev/null || true
    rmdir "$DATA/model/GeneT5" 2>/dev/null || true
else
    echo "  Model data already migrated (or symlink)"
fi

# Logs: move GeneT5-specific logs
if [ -d "$DATA/logs" ] && [ ! -L "$DATA/logs" ]; then
    echo "  Moving logs..."
    [ -d "$DATA/logs/GeneT5" ]  && mv "$DATA"/logs/GeneT5/*  "$DATA"/GeneT5/logs/ 2>/dev/null || true
    [ -d "$DATA/logs/baker" ]   && mv "$DATA"/logs/baker      "$DATA"/GeneT5/logs/baker 2>/dev/null || true
    mv "$DATA"/logs/bake_*.log      "$DATA"/GeneT5/logs/ 2>/dev/null || true
    mv "$DATA"/logs/diffusion_*.log "$DATA"/GeneT5/logs/ 2>/dev/null || true
    mv "$DATA"/logs/mar12_*.log     "$DATA"/GeneT5/logs/ 2>/dev/null || true
else
    echo "  Logs already migrated (or symlink)"
fi

# Symlink shared raw genomes into GeneT5
echo "[3/4] Creating symlinks..."

if [ ! -e "$DATA/GeneT5/raw" ]; then
    ln -s "$DATA/genome/raw" "$DATA/GeneT5/raw"
    echo "  Created: GeneT5/raw -> genome/raw"
else
    echo "  GeneT5/raw already exists"
fi

# Backward-compat symlinks (old paths -> new locations)
if [ -d "$DATA/genome/baked" ] && [ ! -e "$DATA/genome/baked/GeneT5" ]; then
    ln -s "$DATA/GeneT5/baked" "$DATA/genome/baked/GeneT5"
    echo "  Created: genome/baked/GeneT5 -> GeneT5/baked (compat)"
fi

if [ -d "$DATA/model" ] && [ ! -e "$DATA/model/GeneT5" ]; then
    ln -s "$DATA/GeneT5/model" "$DATA/model/GeneT5"
    echo "  Created: model/GeneT5 -> GeneT5/model (compat)"
fi

# Verify
echo ""
echo "[4/4] Verification..."
echo "  GeneT5/raw:   $(readlink -f "$DATA/GeneT5/raw" 2>/dev/null || echo 'MISSING')"
echo "  GeneT5/baked: $(ls "$DATA/GeneT5/baked/" 2>/dev/null | wc -w) items"
echo "  GeneT5/model: $(ls "$DATA/GeneT5/model/" 2>/dev/null | wc -w) items"
echo "  GeneT5/logs:  $(ls "$DATA/GeneT5/logs/"  2>/dev/null | wc -w) items"
echo ""
echo "=== Migration complete ==="
