"""
eval_saboteur.py - 학습된 사보타주를 여러 인간팀 상대로 평가 (학습 없음)

  python sim/eval_saboteur.py --checkpoint <사보타주 체크포인트.pt> [--games 200] [--crew-checkpoint <인간팀 체크포인트.pt>]

상대 (config/TheThing.yaml 기준):
  bot            : 기본 규칙봇 인간팀 (학습 상대)
  bot_perc0.7    : 방 상태를 아는 거리 0.7배
  bot_perc1.5    : 1.5배
  bot_react3     : 손상을 안 뒤 3초 늦게 대응
  bot_nochase    : 함장이 근거 있는 대상을 쫓지 않음
  bot_old        : 이전 규칙봇 (bot_level 0)
  learned_crew   : --crew-checkpoint의 학습된 인간·함장 정책 (공동 학습 실험의 4M 모델)

행동은 학습 때와 같은 확률적 샘플링. 트레이너와 같은 루프(게임마다 env.reset)로 돈다.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fake_mlagents"))
sys.path.insert(0, ROOT)

import bot_ladder  # noqa: E402
from bot_ladder import yaml_params  # noqa: E402
from mappo_trainer import CONFIG, make_actor  # noqa: E402

VARIANTS = {
    "bot": {},
    "bot_perc0.7": {"bot_perception": 0.7},
    "bot_perc1.5": {"bot_perception": 1.5},
    "bot_react3": {"bot_reaction": 3.0},
    "bot_nochase": {"bot_chase": 0.0},
    "bot_old": {"bot_level": 0.0},
}


def load_actors(path, roles):
    ck = torch.load(path, map_location="cpu")
    out = {}
    for role in roles:
        a = make_actor(role, ck.get("action_mask", False))
        a.load_state_dict(ck[f"actor_{role}"])
        a.eval()
        out[role] = a
    return out


def play(params, sab_actor, crew_actors, n, seed):
    """crew_actors가 None이면 인간팀은 규칙봇(평가 모드 frozen_team 2)"""
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

    params = dict(params, bot_team=0.0)
    if crew_actors is None:
        params.update(eval_mode=1.0, frozen_team=2.0)
    elif sab_actor is None:
        params.update(eval_mode=1.0, frozen_team=1.0)     # 규칙봇 사보타주 vs 학습된 인간팀
    else:
        params.update(eval_mode=0.0, frozen_team=0.0)
    pc = EnvironmentParametersChannel()
    for k, v in params.items():
        pc.set_float_parameter(k, float(v))
    os.environ["SIM_SEED"] = str(seed)
    env = UnityEnvironment(side_channels=[pc])
    env.world.env_params = dict(params)
    env.world._reset_state()            # 첫 판부터 같은 규칙
    torch.manual_seed(seed)
    env.reset()
    L = CONFIG["local_obs_dim"]
    roles, term = {}, set()
    ident = lambda o: "captain" if o[1] > 0.5 else ("saboteur" if o[0] > 0.5 else "human")
    start = len(env.game_log)
    while len(env.game_log) - start < n:
        d, t = env.get_steps("TheThing")
        for aid in t.agent_id:
            if int(aid) in roles:
                term.add(int(aid))
        if roles and set(roles).issubset(term):
            roles.clear()
            term.clear()
            env.reset()
            continue
        if len(d) == 0:
            env.step()
            continue
        acts = np.zeros((len(d), 2), dtype=np.int32)
        with torch.no_grad():
            for i, aid in enumerate(d.agent_id):
                aid = int(aid)
                obs = d[aid].obs[0]
                role = roles.setdefault(aid, ident(obs))
                actor = sab_actor if role == "saboteur" else (crew_actors or {}).get(role)
                if actor is None:
                    acts[i] = (8, 0)        # 규칙봇이 조종 (행동 무시됨)
                    continue
                a, _ = actor.get_action(torch.FloatTensor(obs[:L]).unsqueeze(0))
                acts[i] = a.squeeze(0).numpy()
        env.set_actions("TheThing", ActionTuple(discrete=acts))
        env.step()
    return env.game_log[start:start + n]


def summary(log):
    m = lambda k: round(float(np.mean([g.get(k, 0) for g in log])), 2)
    shots = sum(g["shots"] for g in log)
    return {
        "games": len(log),
        "sab_win": round(1 - float(np.mean([g["human_win"] for g in log])), 3),
        "dur": m("duration"), "sab": m("sab_total"), "hidden": m("sab_hidden"), "rep": m("repairs"),
        "cut": m("sab_interrupted"), "cancel": m("sab_cancelled"), "shots": m("shots"),
        "hit": round(sum(g["shots_hit"] for g in log) / shots, 2) if shots else None,
        "destroy": round(float(np.mean(["파괴" in g["reason"] for g in log])), 3),
    }


def _job(a):
    name, params, sab_path, crew_path, n, seed = a
    torch.set_num_threads(1)
    if sab_path == "rulebot" and crew_path:   # 규칙봇 사보타주 vs 학습된 인간팀
        return name, summary(play(params, None, load_actors(crew_path, ["human", "captain"]), n, seed))
    if sab_path in bot_ladder.POLICIES:      # 기준선: 스크립트/규칙봇 사보타주
        log = bot_ladder.play(params, sab_path, n, seed)
        return name, summary([dict(g, sab_hidden=g.get("sab_hidden", 0)) for g in log])
    sab = load_actors(sab_path, ["saboteur"])["saboteur"]
    crew = load_actors(crew_path, ["human", "captain"]) if crew_path else None
    return name, summary(play(params, sab, crew, n, seed))


def evaluate(sab_path, n=200, crew_path=None, seed=123, variants=None, base=None, workers=4):
    base = base if base is not None else yaml_params()
    jobs = [(name, dict(base, **VARIANTS[name]), sab_path, None, n, seed) for name in (variants or VARIANTS)]
    if crew_path:
        jobs.append(("learned_crew", dict(base), sab_path, crew_path, n, seed))
    with Pool(min(workers, len(jobs))) as pool:
        return dict(pool.map(_job, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="체크포인트 .pt, 또는 기준선 이름 (rulebot, camper, ...)")
    ap.add_argument("--crew-checkpoint", default=None)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--only", default="", help="쉼표로 구분한 상대 이름만 (기본: 전부)")
    ap.add_argument("--param", action="append", default=[], help="밸런스 덮어쓰기 key=value")
    args = ap.parse_args()
    base = yaml_params()
    for kv in args.param:
        k, v = kv.split("=", 1)
        base[k] = float(v)
    only = [x for x in args.only.split(",") if x] or None
    res = evaluate(args.checkpoint, args.games, args.crew_checkpoint, args.seed,
                   [v for v in (only or VARIANTS) if v in VARIANTS], base)
    for k, v in res.items():
        print(f"{k:13s}", json.dumps(v, ensure_ascii=False))
