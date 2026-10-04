"""
merge_runs.py - 체크포인트에서 이어 학습한 실행을 앞 실행과 합쳐 하나의 기록으로 만든다

  python sim/merge_runs.py <앞 실행 dir> <이어 학습 dir> <체크포인트 에피소드 수> <출력 dir>

- games.jsonl: 앞 실행의 처음 N게임(N = 체크포인트 에피소드 수) + 이어 학습 게임.
  이어 학습 쪽 agent_decisions에는 앞 실행 N번째 게임의 값을 더해 축을 이어 붙인다.
- TensorBoard 이벤트 파일은 두 실행 것을 모두 복사한다(이어 학습 쪽 step은 이미 이어진 total_steps).
"""
import glob
import json
import os
import shutil
import sys

first, cont, n_ckpt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(os.path.join(out, "runs", "mappo"), exist_ok=True)

with open(os.path.join(first, "games.jsonl"), encoding="utf-8") as f:
    g1 = [json.loads(l) for l in f][:n_ckpt]
with open(os.path.join(cont, "games.jsonl"), encoding="utf-8") as f:
    g2 = [json.loads(l) for l in f]
offset = g1[-1]["agent_decisions"] if g1 else 0
for g in g2:
    g["agent_decisions"] += offset
    g["continued"] = 1
with open(os.path.join(out, "games.jsonl"), "w", encoding="utf-8") as f:
    for g in g1 + g2:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")

for i, d in enumerate((first, cont)):
    for ev in glob.glob(os.path.join(d, "runs", "mappo", "events.*")):
        shutil.copy(ev, os.path.join(out, "runs", "mappo", f"{i}_" + os.path.basename(ev)))
print(f"{out}: {len(g1)} + {len(g2)} games (offset {offset})")
