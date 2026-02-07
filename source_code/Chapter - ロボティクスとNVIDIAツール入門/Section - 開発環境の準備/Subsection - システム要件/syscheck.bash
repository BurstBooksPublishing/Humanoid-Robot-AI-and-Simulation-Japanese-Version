bash
#!/usr/bin/env bash
set -euo pipefail

# 必要なコマンドの存在チェック
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: NVIDIA driver をインストールしてください" >&2; exit 1; }

# GPU 情報表示
echo "=== GPU 情報 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

# CUDA Toolkit チェック
if command -v nvcc >/dev/null; then
  echo "=== CUDA Toolkit 情報 ==="
  nvcc --version
else
  echo "nvcc not found: CUDA Toolkit を確認してください" >&2
fi

# Docker & nvidia-container-toolkit チェック
if command -v docker >/dev/null; then
  echo "=== Docker 情報 ==="
  docker --version
  # GPU サポート確認
  if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "NVIDIA Container Runtime: OK"
    # 最小 GPU テスト
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null \
      && echo "Docker GPU テスト: 成功" \
      || { echo "Docker GPU テスト: 失敗" >&2; }
  else
    echo "NVIDIA Container Runtime: 未検出" >&2
  fi
else
  echo "Docker: 未インストール" >&2
fi

# ROS 2 チェック
if command -v ros2 >/dev/null; then
  echo "=== ROS 2 情報 ==="
  ros2 --version
  # ROS_DISTRO 環境変数チェック
  [[ -n "${ROS_DISTRO:-}" ]] && echo "ROS_DISTRO: $ROS_DISTRO" || echo "ROS_DISTRO: 未設定" >&2
else
  echo "ROS 2: 未検出" >&2
fi

# Python バージョン表示
echo "=== Python 情報 ==="
python3 -c "import sys, platform; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({platform.machine()})')"