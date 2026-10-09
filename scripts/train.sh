#!/bin/bash
# TheThing MAPPO 학습 (mappo_trainer.py, Unity 빌드에 연결)
# 사용법: ./scripts/train.sh [빌드 경로] [총 스텝]
#   예) ./scripts/train.sh "Builds/Windows/My project.exe" 4000000
# 체크포인트: checkpoints/, TensorBoard 로그: runs/mappo/
# (주의) mlagents-learn은 쓰지 않는다. 이 프로젝트는 커스텀 MAPPO 트레이너를 쓴다.

ENV_PATH="${1:-Builds/Windows/My project.exe}"
STEPS="${2:-4000000}"

echo "=== TheThing MAPPO Training ==="
echo "Build: $ENV_PATH"
echo "Steps: $STEPS"
echo "==============================="

python mappo_trainer.py --env-path "$ENV_PATH" --timesteps "$STEPS"
