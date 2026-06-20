#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_GPUS="${GPUS:-1,2,3}"
DEFAULT_GPUS_LABEL="${DEFAULT_GPUS//,/}"
DEFAULT_NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-4096}"

export TASK_NAME="${TASK_NAME:-ATEC-Isaac-TaskD-G1-CrossPitBox-v1.5}"
export RUN_NAME="${RUN_NAME:-cross_pit_box_v1_5_blind_mlp_pbrs_x_gpus_${DEFAULT_GPUS_LABEL}_env${DEFAULT_NUM_ENVS_PER_GPU}}"
export LAUNCH_MODE="${LAUNCH_MODE:-per_rank}"

exec "${SCRIPT_DIR}/train_cross_pit_box_multigpu.sh" "$@"
