"""
bot_ladder.py - 규칙봇 인간팀을 상대로 한 사보타주 전략 사다리 (학습 없이)

  python sim/bot_ladder.py [게임 수] [key=value ...]

사보타주 쪽 (인간팀·함장은 모두 규칙봇):
  random   : 무작위 행동
  camper   : 각자 가장 가까운 방에 붙어 계속 부수기
  focus    : 둘 다 같은 방(첫 사보타주에게 가장 가까운 방)에 붙기
  evasive  : 인간팀이 적은 방을 고르고, 시야에 인간팀이 보이면 부수지 않고 다른 방으로
  rulebot  : 규칙봇 사보타주

config/TheThing.yaml의 environment_parameters에 인자로 준 값을 덮어쓴다.
"""
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thething_sim import SABOTEUR, TheThingWorld, dist  # noqa: E402

POLICIES = ["random", "camper", "focus", "evasive", "rulebot"]


def yaml_params():
    with open(os.path.join(os.path.dirname(HERE), "config", "TheThing.yaml"), encoding="utf-8") as f:
        return {k: float(v) for k, v in yaml.safe_load(f)["environment_parameters"].items()}


def _crew_visible(w, c):
    vis = w.p("vision_range")
    return [o for o in w.alive if o.active and w.role[o.idx] != SABOTEUR and dist(o.pos, c.pos) <= vis]


def scripted_action(w, c, policy, rng):
    if policy == "random":
        return int(rng.integers(0, 9)), int(rng.integers(0, 9))
    sabs = sorted(w.saboteurs, key=lambda s: s.idx)
    k = sabs.index(c) if c in sabs else 0
    if policy == "camper":
        order = sorted(w.rooms, key=lambda r: dist(c.pos, r.pos))
        return order[k % len(order)].idx, 1
    if policy == "focus":
        lead = sabs[0] if sabs else c
        room = getattr(w, "_focus_room", None)
        if room is None or room.health <= 0 or getattr(w, "_focus_game", -1) != w.game_index:
            room = min(w.rooms, key=lambda r: dist(lead.pos, r.pos))
            w._focus_room, w._focus_game = room, w.game_index
        return room.idx, 1
    if policy == "evasive":
        seen = _crew_visible(w, c)
        tgt = getattr(c, "_ev_target", None)
        if getattr(c, "_ev_game", -1) != w.game_index:
            tgt = None
        if tgt is None or seen:
            crew = [o for o in w.alive if o.active and w.role[o.idx] != SABOTEUR]

            def danger(r):
                return sum(1 for o in crew if dist(o.pos, r.pos) <= 25.0)
            # 다른 사보타주와 다른 방, 인간팀이 적고 가까운 방
            taken = {getattr(o, "_ev_target", None) for o in sabs if o is not c}
            cand = [r for r in w.rooms if r not in taken and r.health > 0] or w.rooms
            tgt = min(cand, key=lambda r: (danger(r), dist(c.pos, r.pos) + (100 if (tgt is r and seen) else 0)))
            c._ev_target, c._ev_game = tgt, w.game_index
        return tgt.idx, (0 if seen else 1)
    raise ValueError(policy)


def play(params, policy, n, seed=0):
    log = []
    w = TheThingWorld(seed=seed, stats_sink={}, game_log=log)
    frozen = 3 if policy == "rulebot" else 2
    w.env_params = dict(params, eval_mode=1.0, frozen_team=float(frozen), bot_team=0.0)
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
            if w.role[c.idx] == SABOTEUR and c.use_ml:
                acts[aid] = scripted_action(w, c, policy, rng)
            else:
                acts[aid] = (8, 0)
        out = w.advance(acts)
    return log


def summary(log):
    reasons = collections.Counter()
    for g in log:
        r = g["reason"]
        reasons["방 파괴" if "파괴" in r else r] += 1
    m = lambda k: round(float(np.mean([g.get(k, 0) for g in log])), 2)
    return {
        "sab_win": round(1 - float(np.mean([g["human_win"] for g in log])), 3),
        "dur": m("duration"), "sab": m("sab_total"), "hidden": m("sab_hidden"), "rep": m("repairs"),
        "cut": m("sab_interrupted"), "shots": m("shots"),
        "hit": round(float(np.sum([g["shots_hit"] for g in log]) / max(1, np.sum([g["shots"] for g in log]))), 2),
        "reasons": dict(reasons.most_common(4)),
    }


def _job(a):
    params, pol, n, seed = a
    return pol, summary(play(params, pol, n, seed))


def ladder(params, n, policies=POLICIES, seed=0, pool=None):
    jobs = [(params, p, n, seed) for p in policies]
    res = (pool.map if pool else map)(_job, jobs)
    return dict(res)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    params = yaml_params()
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        params[k] = float(v)
    with Pool(min(4, len(POLICIES))) as pool:
        res = ladder(params, n, pool=pool)
    for pol in POLICIES:
        print(f"{pol:8s}", json.dumps(res[pol], ensure_ascii=False))
