"""
baselines.py - 학습 없이 기준 승률 측정

  random_vs_random : 전원 무작위 행동 (학습 시작점과 같은 조건)
  random_sab_vs_bot: 무작위 사보타주 vs 규칙봇 인간팀  (eval/saboteur_vs_bot_win 의 기준선)
  random_hum_vs_bot: 무작위 인간팀 vs 규칙봇 사보타주  (eval/human_vs_bot_win 의 기준선)
  noshoot_vs_random: 함장 사격만 막은 무작위 (사격 효과 분리)

사용법: python sim/baselines.py [게임 수]
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from thething_sim import TheThingWorld  # noqa: E402


def run(n_games, eval_mode=0, frozen=0, seed=0, no_shoot=False):
    log = []
    w = TheThingWorld(seed=seed, stats_sink={}, game_log=log)
    w.env_params = {"eval_mode": float(eval_mode), "frozen_team": float(frozen)}
    w.load_env_params()
    w._reset_state()                     # 첫 게임부터 모드 적용
    rng = np.random.default_rng(seed)
    w.force_reset()
    out = w.first_exchange()
    while len(log) < n_games:
        acts = {}
        for aid, obs, r, done in out:
            if not done:
                inter = int(rng.integers(0, 9))
                if no_shoot and inter >= 3:
                    inter = 0
                acts[aid] = (int(rng.integers(0, 9)), inter)
        out = w.advance(acts)
    return log


def summarize(name, log):
    hw = np.mean([g["human_win"] for g in log])
    reasons = collections.Counter(g["reason"] for g in log)
    dur = np.mean([g["duration"] for g in log])
    return {"name": name, "games": len(log), "human_win_rate": round(float(hw), 3),
            "mean_duration_s": round(float(dur), 1),
            "reasons": dict(reasons.most_common())}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    results = [
        summarize("random_vs_random", run(n)),
        summarize("random_vs_random_no_shoot", run(n, no_shoot=True)),
        summarize("random_sab_vs_bot_humans", run(n, 1, 2)),
        summarize("random_humans_vs_bot_sab", run(n, 1, 1)),
    ]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
