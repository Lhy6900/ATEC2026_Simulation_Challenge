#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-1,2,3}"
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${SEED:-42}"
TASK_NAME="${TASK_NAME:-ATEC-Isaac-TaskD-G1-CrossPitBox-v1}"
RUN_NAME="${RUN_NAME:-cross_pit_box_v1_gpus_${GPUS//,/}_env${NUM_ENVS_PER_GPU}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"
TMPDIR="${TMPDIR:-${REPO_ROOT}/tmp_isaaclab}"
PYTHONPATH="${PYTHONPATH:-.}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
HEADLESS="${HEADLESS:-1}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-0}"
FAST_EXIT_AFTER_TRAIN="${FAST_EXIT_AFTER_TRAIN:-1}"
GLX_VENDOR="${GLX_VENDOR:-nvidia}"
NV_PRIME_RENDER_OFFLOAD="${NV_PRIME_RENDER_OFFLOAD:-1}"
VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}"
LAUNCH_MODE="${LAUNCH_MODE:-torchrun}"
CLEANUP_EXISTING="${CLEANUP_EXISTING:-1}"
CLEANUP_WAIT_SECONDS="${CLEANUP_WAIT_SECONDS:-20}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-2}"
WAIT_FOR_MASTER="${WAIT_FOR_MASTER:-1}"
MASTER_READY_TIMEOUT_SECONDS="${MASTER_READY_TIMEOUT_SECONDS:-300}"
MASTER_READY_POLL_SECONDS="${MASTER_READY_POLL_SECONDS:-2}"
WAIT_FOR_RANK_SETUP="${WAIT_FOR_RANK_SETUP:-1}"
RANK_SETUP_TIMEOUT_SECONDS="${RANK_SETUP_TIMEOUT_SECONDS:-420}"
RANK_SETUP_POLL_SECONDS="${RANK_SETUP_POLL_SECONDS:-2}"
TORCHRUN_STARTUP_LOCK_SECONDS="${TORCHRUN_STARTUP_LOCK_SECONDS:-60}"
LOG_RANK_OUTPUTS="${LOG_RANK_OUTPUTS:-1}"
SERIALIZE_ISAAC_STARTUP="${SERIALIZE_ISAAC_STARTUP:-1}"
PER_RANK_TORCHRUN_STATE="${PER_RANK_TORCHRUN_STATE:-1}"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y%m%d_%H%M%S)}"
RANK_LOG_DIR="${RANK_LOG_DIR:-${TMPDIR}/cross_pit_box_logs/${RUN_NAME}_${LAUNCH_ID}}"
RANK_STATE_ROOT="${RANK_STATE_ROOT:-${TMPDIR}/cross_pit_box_state/${RUN_NAME}_${LAUNCH_ID}}"
ISAAC_STARTUP_LOCK_FILE="${ISAAC_STARTUP_LOCK_FILE:-${TMPDIR}/cross_pit_box_isaac_startup.lock}"
ISAAC_STARTUP_LOCK_FD=""
ISAAC_STARTUP_LOCK_HELD=0

if [[ -z "${VK_ICD_FILENAMES}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
fi

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_LIST[@]}}"

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "NPROC_PER_NODE must be >= 1" >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE}" -gt "${#GPU_LIST[@]}" ]]; then
  echo "NPROC_PER_NODE (${NPROC_PER_NODE}) cannot exceed number of GPUs in GPUS (${#GPU_LIST[@]})" >&2
  exit 1
fi
if [[ "${LAUNCH_MODE}" != "per_rank" && "${LAUNCH_MODE}" != "torchrun" ]]; then
  echo "LAUNCH_MODE must be 'per_rank' or 'torchrun'" >&2
  exit 1
fi
if [[ "${CLEANUP_EXISTING}" != "0" && "${CLEANUP_EXISTING}" != "1" ]]; then
  echo "CLEANUP_EXISTING must be 0 or 1" >&2
  exit 1
fi
if [[ "${LOG_RANK_OUTPUTS}" != "0" && "${LOG_RANK_OUTPUTS}" != "1" ]]; then
  echo "LOG_RANK_OUTPUTS must be 0 or 1" >&2
  exit 1
fi
if [[ "${SERIALIZE_ISAAC_STARTUP}" != "0" && "${SERIALIZE_ISAAC_STARTUP}" != "1" ]]; then
  echo "SERIALIZE_ISAAC_STARTUP must be 0 or 1" >&2
  exit 1
fi
if [[ "${PER_RANK_TORCHRUN_STATE}" != "0" && "${PER_RANK_TORCHRUN_STATE}" != "1" ]]; then
  echo "PER_RANK_TORCHRUN_STATE must be 0 or 1" >&2
  exit 1
fi
if [[ "${SERIALIZE_ISAAC_STARTUP}" == "1" ]] && ! command -v flock >/dev/null 2>&1; then
  echo "SERIALIZE_ISAAC_STARTUP=1 requires flock to be available" >&2
  exit 1
fi
if [[ "${WAIT_FOR_MASTER}" != "0" && "${WAIT_FOR_MASTER}" != "1" ]]; then
  echo "WAIT_FOR_MASTER must be 0 or 1" >&2
  exit 1
fi
if [[ "${MASTER_READY_TIMEOUT_SECONDS}" -lt 1 ]]; then
  echo "MASTER_READY_TIMEOUT_SECONDS must be >= 1" >&2
  exit 1
fi
if [[ "${MASTER_READY_POLL_SECONDS}" -lt 1 ]]; then
  echo "MASTER_READY_POLL_SECONDS must be >= 1" >&2
  exit 1
fi
if [[ "${WAIT_FOR_RANK_SETUP}" != "0" && "${WAIT_FOR_RANK_SETUP}" != "1" ]]; then
  echo "WAIT_FOR_RANK_SETUP must be 0 or 1" >&2
  exit 1
fi
if [[ "${RANK_SETUP_TIMEOUT_SECONDS}" -lt 1 ]]; then
  echo "RANK_SETUP_TIMEOUT_SECONDS must be >= 1" >&2
  exit 1
fi
if [[ "${RANK_SETUP_POLL_SECONDS}" -lt 1 ]]; then
  echo "RANK_SETUP_POLL_SECONDS must be >= 1" >&2
  exit 1
fi
if [[ "${TORCHRUN_STARTUP_LOCK_SECONDS}" -lt 1 ]]; then
  echo "TORCHRUN_STARTUP_LOCK_SECONDS must be >= 1" >&2
  exit 1
fi

mkdir -p "${TMPDIR}"
if [[ "${LOG_RANK_OUTPUTS}" == "1" ]]; then
  mkdir -p "${RANK_LOG_DIR}"
fi
mkdir -p "${RANK_STATE_ROOT}"

terminate_pids() {
  local pids=("$@")
  local remaining=()
  if [[ ${#pids[@]} -eq 0 ]]; then
    return 0
  fi

  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 5

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      remaining+=("${pid}")
    fi
  done

  if [[ ${#remaining[@]} -gt 0 ]]; then
    kill -KILL "${remaining[@]}" 2>/dev/null || true
  fi
}

find_existing_training_pids() {
  local pid cmdline
  for pid in /proc/[0-9]*; do
    pid="${pid#/proc/}"
    [[ "${pid}" == "$$" ]] && continue
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    case "${cmdline}" in
      *"python -u scripts/rsl_rl/train.py --task ATEC-Isaac-TaskD-G1-CrossPitBox"*--distributed*|*"python scripts/rsl_rl/train.py --task ATEC-Isaac-TaskD-G1-CrossPitBox"*--distributed*)
        printf '%s\n' "${pid}"
        ;;
    esac
  done
}

cleanup_existing_training() {
  if [[ "${CLEANUP_EXISTING}" != "1" ]]; then
    return 0
  fi

  mapfile -t existing_pids < <(find_existing_training_pids)
  if [[ ${#existing_pids[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Stopping existing CrossPitBox training processes before launch: ${existing_pids[*]}"
  terminate_pids "${existing_pids[@]}"

  local waited=0
  while [[ "${waited}" -lt "${CLEANUP_WAIT_SECONDS}" ]]; do
    mapfile -t remaining_pids < <(find_existing_training_pids)
    if [[ ${#remaining_pids[@]} -eq 0 ]]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  mapfile -t remaining_pids < <(find_existing_training_pids)
  if [[ ${#remaining_pids[@]} -gt 0 ]]; then
    echo "Unable to stop existing CrossPitBox training processes: ${remaining_pids[*]}" >&2
    return 1
  fi
}

acquire_isaac_startup_lock() {
  if [[ "${SERIALIZE_ISAAC_STARTUP}" != "1" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${ISAAC_STARTUP_LOCK_FILE}")"
  exec {ISAAC_STARTUP_LOCK_FD}>"${ISAAC_STARTUP_LOCK_FILE}"
  echo "Waiting for Isaac startup lock: ${ISAAC_STARTUP_LOCK_FILE}"
  flock "${ISAAC_STARTUP_LOCK_FD}"
  ISAAC_STARTUP_LOCK_HELD=1
  echo "Acquired Isaac startup lock"
}

release_isaac_startup_lock() {
  if [[ "${ISAAC_STARTUP_LOCK_HELD}" != "1" ]]; then
    return 0
  fi

  flock -u "${ISAAC_STARTUP_LOCK_FD}" || true
  exec {ISAAC_STARTUP_LOCK_FD}>&- || true
  ISAAC_STARTUP_LOCK_HELD=0
  echo "Released Isaac startup lock"
}

TRAIN_ARGS=(
  scripts/rsl_rl/train.py
  --task "${TASK_NAME}"
  --distributed
  --num_envs "${NUM_ENVS_PER_GPU}"
  --max_iterations "${MAX_ITERATIONS}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
)

if [[ "${ENABLE_CAMERAS}" == "1" ]]; then
  TRAIN_ARGS+=(--enable_cameras)
fi

if [[ "${FAST_EXIT_AFTER_TRAIN}" == "1" ]]; then
  TRAIN_ARGS+=(--fast_exit)
fi

if [[ "${HEADLESS}" == "1" ]]; then
  TRAIN_ARGS+=(--headless)
fi

TRAIN_ARGS+=("$@")

echo "Repo: ${REPO_ROOT}"
echo "GPUs: ${GPUS}"
echo "Processes: ${NPROC_PER_NODE}"
echo "Launch mode: ${LAUNCH_MODE}"
echo "Task: ${TASK_NAME}"
echo "Num envs per GPU/rank: ${NUM_ENVS_PER_GPU}"
echo "Run name: ${RUN_NAME}"
echo "TMPDIR: ${TMPDIR}"
echo "Cleanup existing: ${CLEANUP_EXISTING}"
echo "PyTorch CUDA alloc conf: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Headless: ${HEADLESS}"
echo "Enable cameras: ${ENABLE_CAMERAS}"
echo "Fast exit after train: ${FAST_EXIT_AFTER_TRAIN}"
echo "GLX vendor: ${GLX_VENDOR}"
echo "Wait for master: ${WAIT_FOR_MASTER}"
echo "Master ready timeout seconds: ${MASTER_READY_TIMEOUT_SECONDS}"
echo "Master ready poll seconds: ${MASTER_READY_POLL_SECONDS}"
echo "Wait for rank setup: ${WAIT_FOR_RANK_SETUP}"
echo "Rank setup timeout seconds: ${RANK_SETUP_TIMEOUT_SECONDS}"
echo "Rank setup poll seconds: ${RANK_SETUP_POLL_SECONDS}"
echo "Torchrun startup lock seconds: ${TORCHRUN_STARTUP_LOCK_SECONDS}"
echo "Log rank outputs: ${LOG_RANK_OUTPUTS}"
echo "Serialize Isaac startup: ${SERIALIZE_ISAAC_STARTUP}"
echo "Per-rank torchrun Isaac state: ${PER_RANK_TORCHRUN_STATE}"
if [[ "${SERIALIZE_ISAAC_STARTUP}" == "1" ]]; then
  echo "Isaac startup lock file: ${ISAAC_STARTUP_LOCK_FILE}"
fi
if [[ "${LOG_RANK_OUTPUTS}" == "1" ]]; then
  echo "Rank log dir: ${RANK_LOG_DIR}"
fi
echo "Rank state root: ${RANK_STATE_ROOT}"
if [[ -n "${VK_ICD_FILENAMES}" ]]; then
  echo "Vulkan ICD: ${VK_ICD_FILENAMES}"
fi
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Command template:"
if [[ "${LAUNCH_MODE}" == "torchrun" ]]; then
  printf ' CUDA_VISIBLE_DEVICES=%q' "${GPUS}"
  printf ' %q' "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_ARGS[@]}"
else
  for ((rank = 0; rank < NPROC_PER_NODE; rank++)); do
    printf ' [rank %d] CUDA_VISIBLE_DEVICES=%q RANK=%d LOCAL_RANK=0 WORLD_SIZE=%d %q' \
      "${rank}" "${GPU_LIST[$rank]}" "${rank}" "${NPROC_PER_NODE}" "${PYTHON_BIN}"
    printf ' %q' "${TRAIN_ARGS[@]}"
    echo
  done
fi
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  if [[ "${LAUNCH_MODE}" == "torchrun" ]]; then
    echo "torchrun will map LOCAL_RANK 0..$((NPROC_PER_NODE - 1)) onto CUDA_VISIBLE_DEVICES=${GPUS}."
    for ((rank = 0; rank < NPROC_PER_NODE; rank++)); do
      echo "Rank ${rank} -> logical cuda:${rank} -> physical GPU ${GPU_LIST[$rank]}"
    done
  else
    echo "per_rank will launch one process per GPU with only that physical GPU visible to Isaac/Kit."
    for ((rank = 0; rank < NPROC_PER_NODE; rank++)); do
      echo "Rank ${rank} -> CUDA_VISIBLE_DEVICES=${GPU_LIST[$rank]} RANK=${rank} LOCAL_RANK=0 WORLD_SIZE=${NPROC_PER_NODE}"
    done
  fi
  exit 0
fi

cleanup_existing_training

if [[ "${LAUNCH_MODE}" == "torchrun" ]]; then
  mkdir -p \
    "${RANK_STATE_ROOT}/torchrun/tmp" \
    "${RANK_STATE_ROOT}/torchrun/cache" \
    "${RANK_STATE_ROOT}/torchrun/config" \
    "${RANK_STATE_ROOT}/torchrun/data" \
    "${RANK_STATE_ROOT}/torchrun/omniverse-cache" \
    "${RANK_STATE_ROOT}/torchrun/omniverse-data" \
    "${RANK_STATE_ROOT}/torchrun/omniverse-logs"
  acquire_isaac_startup_lock
  CUDA_VISIBLE_DEVICES="${GPUS}" \
    TMPDIR="${RANK_STATE_ROOT}/torchrun/tmp" \
    XDG_CACHE_HOME="${RANK_STATE_ROOT}/torchrun/cache" \
    XDG_CONFIG_HOME="${RANK_STATE_ROOT}/torchrun/config" \
    XDG_DATA_HOME="${RANK_STATE_ROOT}/torchrun/data" \
    OMNI_USER_CACHE_DIR="${RANK_STATE_ROOT}/torchrun/omniverse-cache" \
    OMNI_USER_DATA_DIR="${RANK_STATE_ROOT}/torchrun/omniverse-data" \
    OMNI_USER_LOG_DIR="${RANK_STATE_ROOT}/torchrun/omniverse-logs" \
    ATEC_PER_RANK_OMNI_STATE="${PER_RANK_TORCHRUN_STATE}" \
    ATEC_RANK_STATE_ROOT="${RANK_STATE_ROOT}/torchrun_ranks" \
    PYTHONPATH="${PYTHONPATH}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    __GLX_VENDOR_LIBRARY_NAME="${GLX_VENDOR}" \
    __NV_PRIME_RENDER_OFFLOAD="${NV_PRIME_RENDER_OFFLOAD}" \
    VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${TRAIN_ARGS[@]}" &
  torchrun_pid="$!"
  if [[ "${SERIALIZE_ISAAC_STARTUP}" == "1" ]]; then
    waited=0
    while [[ "${waited}" -lt "${TORCHRUN_STARTUP_LOCK_SECONDS}" ]]; do
      if ! kill -0 "${torchrun_pid}" 2>/dev/null; then
        release_isaac_startup_lock
        wait "${torchrun_pid}"
        exit $?
      fi
      sleep 1
      waited=$((waited + 1))
    done
    release_isaac_startup_lock
  fi
  wait "${torchrun_pid}"
  exit $?
fi

PIDS=()
cleanup() {
  local status=$?
  release_isaac_startup_lock
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    terminate_pids "${PIDS[@]}"
  fi
  exit "${status}"
}
trap cleanup INT TERM EXIT

fail_launch() {
  local status="${1:-1}"
  release_isaac_startup_lock
  terminate_pids "${PIDS[@]}"
  trap - INT TERM EXIT
  exit "${status}"
}

launch_rank() {
  local rank="$1"
  local rank_log="${RANK_LOG_DIR}/rank${rank}.log"
  local rank_state_dir="${RANK_STATE_ROOT}/rank${rank}"
  mkdir -p \
    "${rank_state_dir}/tmp" \
    "${rank_state_dir}/cache" \
    "${rank_state_dir}/config" \
    "${rank_state_dir}/data" \
    "${rank_state_dir}/omniverse-cache" \
    "${rank_state_dir}/omniverse-data" \
    "${rank_state_dir}/omniverse-logs"
  if [[ "${LOG_RANK_OUTPUTS}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" \
      TMPDIR="${rank_state_dir}/tmp" \
      XDG_CACHE_HOME="${rank_state_dir}/cache" \
      XDG_CONFIG_HOME="${rank_state_dir}/config" \
      XDG_DATA_HOME="${rank_state_dir}/data" \
      OMNI_USER_CACHE_DIR="${rank_state_dir}/omniverse-cache" \
      OMNI_USER_DATA_DIR="${rank_state_dir}/omniverse-data" \
      OMNI_USER_LOG_DIR="${rank_state_dir}/omniverse-logs" \
      PYTHONPATH="${PYTHONPATH}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      __GLX_VENDOR_LIBRARY_NAME="${GLX_VENDOR}" \
      __NV_PRIME_RENDER_OFFLOAD="${NV_PRIME_RENDER_OFFLOAD}" \
      VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
      MASTER_ADDR="${MASTER_ADDR}" \
      MASTER_PORT="${MASTER_PORT}" \
      WORLD_SIZE="${NPROC_PER_NODE}" \
      RANK="${rank}" \
      LOCAL_RANK="0" \
      "${PYTHON_BIN}" -u "${TRAIN_ARGS[@]}" > >(tee "${rank_log}") 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" \
      TMPDIR="${rank_state_dir}/tmp" \
      XDG_CACHE_HOME="${rank_state_dir}/cache" \
      XDG_CONFIG_HOME="${rank_state_dir}/config" \
      XDG_DATA_HOME="${rank_state_dir}/data" \
      OMNI_USER_CACHE_DIR="${rank_state_dir}/omniverse-cache" \
      OMNI_USER_DATA_DIR="${rank_state_dir}/omniverse-data" \
      OMNI_USER_LOG_DIR="${rank_state_dir}/omniverse-logs" \
      PYTHONPATH="${PYTHONPATH}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      __GLX_VENDOR_LIBRARY_NAME="${GLX_VENDOR}" \
      __NV_PRIME_RENDER_OFFLOAD="${NV_PRIME_RENDER_OFFLOAD}" \
      VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
      MASTER_ADDR="${MASTER_ADDR}" \
      MASTER_PORT="${MASTER_PORT}" \
      WORLD_SIZE="${NPROC_PER_NODE}" \
      RANK="${rank}" \
      LOCAL_RANK="0" \
      "${PYTHON_BIN}" -u "${TRAIN_ARGS[@]}" &
  fi
}

master_store_listening() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH 2>/dev/null | awk -v port=":${MASTER_PORT}" '
      $4 ~ port "$" { found = 1 }
      END { exit(found ? 0 : 1) }
    '
    return $?
  fi

  (exec 3<>"/dev/tcp/${MASTER_ADDR}/${MASTER_PORT}") >/dev/null 2>&1
}

wait_for_master_store() {
  if [[ "${WAIT_FOR_MASTER}" != "1" || "${NPROC_PER_NODE}" -le 1 ]]; then
    return 0
  fi

  echo "Waiting for rank 0 TCPStore on ${MASTER_ADDR}:${MASTER_PORT} ..."
  local waited=0
  while [[ "${waited}" -lt "${MASTER_READY_TIMEOUT_SECONDS}" ]]; do
    if master_store_listening; then
      echo "Rank 0 TCPStore is ready on ${MASTER_ADDR}:${MASTER_PORT}"
      return 0
    fi
    if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
      echo "Rank 0 exited before opening the TCPStore on ${MASTER_ADDR}:${MASTER_PORT}" >&2
      return 1
    fi
    sleep "${MASTER_READY_POLL_SECONDS}"
    waited=$((waited + MASTER_READY_POLL_SECONDS))
  done

  echo "Timed out waiting for rank 0 TCPStore on ${MASTER_ADDR}:${MASTER_PORT}" >&2
  return 1
}

rank_setup_completed() {
  local rank="$1"
  local rank_log="${RANK_LOG_DIR}/rank${rank}.log"

  [[ "${LOG_RANK_OUTPUTS}" == "1" ]] || return 0
  [[ -r "${rank_log}" ]] || return 1
  grep -Fq "[INFO]: Completed setting up the environment" "${rank_log}"
}

wait_for_rank_setup() {
  local rank="$1"

  if [[ "${WAIT_FOR_RANK_SETUP}" != "1" || "${LOG_RANK_OUTPUTS}" != "1" ]]; then
    return 0
  fi

  echo "Waiting for rank ${rank} environment setup to complete ..."
  local waited=0
  while [[ "${waited}" -lt "${RANK_SETUP_TIMEOUT_SECONDS}" ]]; do
    if rank_setup_completed "${rank}"; then
      echo "Rank ${rank} completed environment setup"
      return 0
    fi
    if ! kill -0 "${PIDS[$rank]}" 2>/dev/null; then
      echo "Rank ${rank} exited before completing environment setup" >&2
      return 1
    fi
    sleep "${RANK_SETUP_POLL_SECONDS}"
    waited=$((waited + RANK_SETUP_POLL_SECONDS))
  done

  echo "Timed out waiting for rank ${rank} environment setup" >&2
  return 1
}

acquire_isaac_startup_lock

launch_rank 0
PIDS+=("$!")

if ! wait_for_rank_setup "0"; then
  fail_launch 1
fi

if ! wait_for_master_store; then
  fail_launch 1
fi

for ((rank = 1; rank < NPROC_PER_NODE; rank++)); do
  if [[ "${STARTUP_STAGGER_SECONDS}" != "0" ]]; then
    sleep "${STARTUP_STAGGER_SECONDS}"
  fi
  launch_rank "${rank}"
  PIDS+=("$!")
  if ! wait_for_rank_setup "${rank}"; then
    fail_launch 1
  fi
done

release_isaac_startup_lock

FAILED=0
remaining=${#PIDS[@]}
while [[ "${remaining}" -gt 0 ]]; do
  if ! wait -n; then
    FAILED=1
    terminate_pids "${PIDS[@]}"
    break
  fi
  remaining=$((remaining - 1))
done

trap - INT TERM EXIT
exit "${FAILED}"
