]; then
  echo "Isaac Sim not found at ${ISAAC_SIM_DIR}" >&2
  exit 2
fi

# ヘッドレス起動（引数があれば渡す）
exec "${ISAAC_SIM_DIR}/isaac-sim.sh" --headless --no-window "$@"