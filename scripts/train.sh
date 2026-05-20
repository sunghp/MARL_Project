#!/bin/bash
# TheThing MAPPO 학습 시작
# 사용법: ./scripts/train.sh [run_id] [config] [num_envs]

RUN_ID="${1:-TheThing_v1}"
CONFIG="${2:-TheThing}"
NUM_ENVS="${3:-1}"

echo "=== TheThing MAPPO Training ==="
echo "Run ID:   $RUN_ID"
echo "Config:   config/${CONFIG}.yaml"
echo "Num Envs: $NUM_ENVS"
echo "==============================="

mlagents-learn "config/${CONFIG}.yaml" \
    --run-id="$RUN_ID" \
    --num-envs="$NUM_ENVS" \
    --results-dir=results \
    --force
