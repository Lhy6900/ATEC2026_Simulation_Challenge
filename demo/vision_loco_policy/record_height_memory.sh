#!/bin/bash
set -e

export ATEC_VISION_RECORD_HEIGHT="${ATEC_VISION_RECORD_HEIGHT:-1}"
export ATEC_VISION_HEIGHT_MEMORY="${ATEC_VISION_HEIGHT_MEMORY:-vision_height_memory.npz}"
export ATEC_VISION_RECORD_START="${ATEC_VISION_RECORD_START:-9.2}"
export ATEC_VISION_RECORD_END="${ATEC_VISION_RECORD_END:-13.0}"
export ATEC_BOXPUSH_PLANNER_CHECKPOINT="${ATEC_BOXPUSH_PLANNER_CHECKPOINT:-boxpush_planner_v2.pt}"

PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-Record-G1 --enable_cameras --debug
