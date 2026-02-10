#!/bin/bash
#
# One-time NFS setup for two-Spark shared filesystem
#
# Run on Spark 1 HOST (outside Docker):
#   sudo bash /home/cg666/Code/GeneT5/bin/setup-nfs.sh
#
# After running on Spark 1, the script SSHes to Spark 2 to configure mounts

set -e

SPARK1_IP="192.168.100.10"
SPARK2_IP="192.168.100.11"
SPARK2_USER="cg666"

echo "============================================"
echo "  GeneT5 Two-Spark NFS Setup"
echo "============================================"
echo ""

# ── Step 1: Install NFS server on Spark 1 ──
echo "[spark-1] Installing NFS server..."
apt-get -y install nfs-kernel-server > /dev/null 2>&1

# ── Step 2: Configure exports (rw + no_root_squash for Docker) ──
echo "[spark-1] Configuring NFS exports..."

# Back up existing exports
cp /etc/exports /etc/exports.bak 2>/dev/null || true

cat > /etc/exports << EOF
# GeneT5 shared filesystem between two Sparks
/home/cg666/Data  ${SPARK2_IP}/32(rw,sync,no_subtree_check,no_root_squash)
/home/cg666/Code  ${SPARK2_IP}/32(rw,sync,no_subtree_check,no_root_squash)
EOF

# ── Step 3: Apply and restart NFS ──
echo "[spark-1] Restarting NFS server..."
exportfs -arv
systemctl restart nfs-kernel-server
systemctl enable nfs-kernel-server

echo "[spark-1] NFS exports active:"
exportfs -v
echo ""

# ── Step 4: Configure Spark 2 mounts ──
echo "[spark-2] Configuring NFS mounts..."

ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_IP} bash -s << 'REMOTE'
set -e

# Remove old local Code directory (was a separate git clone)
if [ -d /home/cg666/Code/GeneT5/.git ] && ! mountpoint -q /home/cg666/Code; then
    echo "  Removing old local Code copy..."
    rm -rf /home/cg666/Code/GeneT5
    mkdir -p /home/cg666/Code
fi

# Unmount old read-only Data mount if present
if mountpoint -q /home/cg666/Data; then
    echo "  Unmounting old Data mount..."
    sudo umount /home/cg666/Data || sudo umount -l /home/cg666/Data
fi

# Create mount points
mkdir -p /home/cg666/Data /home/cg666/Code

# Mount shared filesystems
echo "  Mounting Data (rw) from Spark 1..."
sudo mount -t nfs 192.168.100.10:/home/cg666/Data /home/cg666/Data -o rw,hard,timeo=600,retrans=2

echo "  Mounting Code (rw) from Spark 1..."
sudo mount -t nfs 192.168.100.10:/home/cg666/Code /home/cg666/Code -o rw,hard,timeo=600,retrans=2

# Verify
echo ""
echo "  Verifying mounts:"
mount | grep 192.168.100.10

# Test write access
touch /home/cg666/Data/genome/baked/.nfs_write_test && rm /home/cg666/Data/genome/baked/.nfs_write_test
echo "  Data: writable OK"

touch /home/cg666/Code/GeneT5/.nfs_write_test && rm /home/cg666/Code/GeneT5/.nfs_write_test
echo "  Code: writable OK"

# Persist in fstab
if ! grep -q "192.168.100.10:/home/cg666/Data" /etc/fstab; then
    echo "" | sudo tee -a /etc/fstab > /dev/null
    echo "# GeneT5 shared filesystem from Spark 1" | sudo tee -a /etc/fstab > /dev/null
    echo "192.168.100.10:/home/cg666/Data  /home/cg666/Data  nfs  rw,hard,timeo=600,retrans=2,nofail,_netdev  0  0" | sudo tee -a /etc/fstab > /dev/null
    echo "192.168.100.10:/home/cg666/Code  /home/cg666/Code  nfs  rw,hard,timeo=600,retrans=2,nofail,_netdev  0  0" | sudo tee -a /etc/fstab > /dev/null
    echo "  Added to /etc/fstab for persistence"
fi
REMOTE

echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "  Shared from Spark 1 -> Spark 2:"
echo "    /home/cg666/Data  (raw, baked, model, logs)"
echo "    /home/cg666/Code  (GeneT5 source)"
echo ""
echo "  Both Sparks now see the same files."
echo "  Restart the gt5 container on Spark 2:"
echo "    ssh ${SPARK2_USER}@${SPARK2_IP}"
echo "    cd /home/cg666/Code/GeneT5"
echo "    bash start-worker.sh"
echo "============================================"
