"""
balance_check.py - 스크립트 사보타주("가장 가까운 방 하나에 붙어서 계속 부수기") vs 규칙봇 인간팀

학습된 사보타주 정책이 찾은 전략이 게임 규칙상 원래 필승인지 확인한다.
  python sim/balance_check.py [게임 수]
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from thething_sim import SABOTEUR, TheThingWorld, dist  # noqa: E402


def run(n, mode):
    log = []
    w = TheThingWorld(seed=0, stats_sink={}, game_log=log)
    w.env_params = {"eval_mode": 1.0, "frozen_team": 2.0}   # 인간팀 = 규칙봇
    w.load_env_params()
    w._reset_state()
    w.force_reset()
    out = w.first_exchange()
    while len(log) < n:
        acts = {}
        by_id = {c.agent.episode_id: c for c in w.chars}
        for aid, obs, r, done in out:
            if done or aid not in by_id:
                continue
            c = by_id[aid]
            if w.role[c.idx] != SABOTEUR:
                acts[aid] = (8, 0)
                continue
            sabs = sorted(w.saboteurs, key=lambda s: s.idx)
            if mode == "same_room":
                room = min(w.rooms, key=lambda r: dist(w.saboteurs[0].pos if w.saboteurs else c.pos, r.pos))
            else:   # 각자 다른 가까운 방
                order = sorted(w.rooms, key=lambda r: dist(c.pos, r.pos))
                k = sabs.index(c) if c in sabs else 0
                room = order[k % len(order)]
            acts[aid] = (room.idx, 1)
        out = w.advance(acts)
    return log


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for mode in ("own_room", "same_room"):
        log = run(n, mode)
        print(mode, "사보타주 승률", round(1 - np.mean([g["human_win"] for g in log]), 3),
              "평균 길이", round(float(np.mean([g["duration"] for g in log])), 1),
              "인간 수리/게임", round(float(np.mean([g["repairs"] for g in log])), 2),
              dict(collections.Counter(("방 파괴" if "파괴" in g["reason"] else g["reason"]) for g in log)))
