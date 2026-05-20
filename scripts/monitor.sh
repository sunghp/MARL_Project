#!/bin/bash
# TensorBoard 모니터링
# 브라우저에서 http://localhost:6006 접속

echo "=== TensorBoard ==="
echo "http://localhost:6006"
echo "===================="

tensorboard --logdir=results --port=6006
