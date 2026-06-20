#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_GPUS="${GPUS:-5,6,7}"
DEFAULT_GPUS_LABEL="${DEFAULT_GPUS//,/}"
DEFAULT_NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-4096}"

export TASK_NAME="${TASK_NAME:-ATEC-Isaac-TaskD-G1-CrossPitBox-v2.5}"
export RUN_NAME="${RUN_NAME:-cross_pit_box_v2_5_blind_mlp_pbrs_x_gpus_${DEFAULT_GPUS_LABEL}_env${DEFAULT_NUM_ENVS_PER_GPU}}"
export GPUS="${DEFAULT_GPUS}"
export MASTER_PORT="${MASTER_PORT:-29592}"
export CLEANUP_EXISTING="${CLEANUP_EXISTING:-0}"
export LAUNCH_MODE="${LAUNCH_MODE:-torchrun}"

exec "${SCRIPT_DIR}/train_cross_pit_box_multigpu.sh" "$@"
