#!/bin/bash
# 중단된 학습 재개 (체크포인트에서 이어서)
# 사용법: ./scripts/train_resume.sh <체크포인트.pt> [빌드 경로] [총 스텝]
#   예) ./scripts/train_resume.sh checkpoints/mappo_interrupted.pt
# Ctrl+C로 멈추면 checkpoints/mappo_interrupted.pt가 저장된다. 총 스텝은 "누적" 기준이다.

CKPT="${1:?체크포인트 경로를 지정하세요}"
ENV_PATH="${2:-Builds/Windows/My project.exe}"
STEPS="${3:-4000000}"

echo "=== Resuming Training: $CKPT ==="

python mappo_trainer.py --load "$CKPT" --env-path "$ENV_PATH" --timesteps "$STEPS"
