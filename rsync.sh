#!/usr/bin/env bash
# 把工作区同步到远端 /home/<remote_user>/RCIA_Benmao_Vision
# build/ install/ log/ 是 colcon 产物, 在远端自己重建, 不同步
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
echo "脚本目录: $SCRIPT_DIR"
echo "工作区目录: $WORK_DIR"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <remote_user> <remote_ip>"
    echo "Example:"
    echo "  $0 nvidiaa 192.168.10.100"
    exit 1
fi

REMOTE_USER="$1"
REMOTE_IP="$2"
TARGET_PATH="/home/${REMOTE_USER}/RCIA_Benmao_Vision"

rsync -avz \
    --exclude='.cache/' \
    --exclude='.vscode/' \
    --exclude='.git/' \
    --exclude='.codex' \
    --exclude='.agents' \
    --exclude='build/' \
    --exclude='install/' \
    --exclude='log/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='*.pyc' \
    --exclude='models/m3t/*.bin'\
    "${WORK_DIR}/" \
    "${REMOTE_USER}@${REMOTE_IP}:${TARGET_PATH}/"
