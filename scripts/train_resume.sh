#!/bin/bash
# 중단된 학습 재개
# 사용법: ./scripts/train_resume.sh [run_id] [config] [num_envs]

RUN_ID="${1:-TheThing_v1}"
CONFIG="${2:-TheThing}"
NUM_ENVS="${3:-1}"

echo "=== Resuming Training: $RUN_ID ==="

mlagents-learn "config/${CONFIG}.yaml" \
    --run-id="$RUN_ID" \
    --num-envs="$NUM_ENVS" \
    --results-dir=results \
    --resume
