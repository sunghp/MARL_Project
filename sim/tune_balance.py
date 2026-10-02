"""
tune_balance.py - 밸런스 설정별 매치업 승률 비교 (학습 없이)

매치업:
  camper_vs_bot  : 스크립트 사보타주(각자 가장 가까운 방에 붙어 계속 부수기) vs 규칙봇 인간팀
  bot_vs_bot     : 양쪽 모두 규칙봇
  rand_sab_vs_bot: 무작위 사보타주 vs 규칙봇 인간팀
  rand_hum_vs_bot: 무작위 인간팀 vs 규칙봇 사보타주
  rand_vs_rand   : 전원 무작위

  python sim/tune_balance.py [게임 수]
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from thething_sim import SABOTEUR, TheThingWorld, dist  # noqa: E402

PRESETS = {
    "current": {},
    # 채택: config/TheThing.yaml에 반영된 값
    "v1_chosen": {
        "room_destroy_loss": 1.0,
        "alert_threshold": 70.0,
        "repair_interrupts_sabotage": 1.0,
        "shoot_grace_time": 10.0,
    },
    # 방 파괴 승리까지 없애면 사보타주가 100초 안에 평균 안정도를 못 떨어뜨려 전패 (과보정)
    "no_room_destroy": {
        "room_destroy_loss": 0.0,
        "alert_threshold": 70.0,
        "repair_interrupts_sabotage": 1.0,
        "shoot_grace_time": 10.0,
    },
    # 알림 50%: 대응 시간이 부족해 여전히 사보타주 필승
    "alert50": {
        "room_destroy_loss": 1.0,
        "alert_threshold": 50.0,
        "repair_interrupts_sabotage": 1.0,
        "shoot_grace_time": 10.0,
    },
}


def play(n, params, frozen, policy, seed=0):
    log = []
    w = TheThingWorld(seed=seed, stats_sink={}, game_log=log)
    w.env_params = dict(params, eval_mode=1.0 if frozen else 0.0, frozen_team=float(frozen))
    w.load_env_params()
    w._reset_state()
    rng = np.random.default_rng(seed)
    w.force_reset()
    out = w.first_exchange()
    while len(log) < n:
        acts = {}
        by_id = {c.agent.episode_id: c for c in w.chars}
        for aid, obs, r, done in out:
            if done or aid not in by_id:
                continue
            c = by_id[aid]
            if policy == "camper" and w.role[c.idx] == SABOTEUR:
                sabs = sorted(w.saboteurs, key=lambda s: s.idx)
                order = sorted(w.rooms, key=lambda r: dist(c.pos, r.pos))
                k = sabs.index(c) if c in sabs else 0
                acts[aid] = (order[k % len(order)].idx, 1)
            else:
                acts[aid] = (int(rng.integers(0, 9)), int(rng.integers(0, 9)))
        out = w.advance(acts)
    return log


def summary(log):
    reasons = collections.Counter()
    for g in log:
        r = g["reason"]
        reasons["방 파괴" if "파괴" in r else r] += 1
    return {
        "sab_win": round(1 - float(np.mean([g["human_win"] for g in log])), 3),
        "dur": round(float(np.mean([g["duration"] for g in log])), 1),
        "sab": round(float(np.mean([g["sab_total"] for g in log])), 1),
        "rep": round(float(np.mean([g["repairs"] for g in log])), 1),
        "cut": round(float(np.mean([g.get("sab_interrupted", 0) for g in log])), 1),
        "reasons": dict(reasons.most_common(4)),
    }


MATCHUPS = [
    ("camper_vs_bot", 2, "camper"),
    ("bot_vs_bot", 3, "random"),
    ("rand_sab_vs_bot", 2, "random"),
    ("rand_hum_vs_bot", 1, "random"),
    ("rand_vs_rand", 0, "random"),
]


def evaluate(params, n):
    return {name: summary(play(n, params, fr, pol)) for name, fr, pol in MATCHUPS}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    names = sys.argv[2:] or list(PRESETS)
    for name in names:
        res = evaluate(PRESETS[name], n)
        print(json.dumps({"preset": name, "params": PRESETS[name], **res}, ensure_ascii=False))
