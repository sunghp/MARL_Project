"""
inspect_shots.py - 학습된 체크포인트로 게임을 돌려 함장 사격을 하나하나 기록

  python sim/inspect_shots.py <checkpoint.pt> [게임 수] [--param k=v ...]
사격마다 대상 역할, 함장 관측의 목격 플래그, 사격 시각, 대상과의 거리, 대상이 수리/부수기 중이었는지 출력.
"""
import collections
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fake_mlagents"))
sys.path.insert(0, os.path.dirname(HERE))
import thething_sim as ts  # noqa: E402
from mappo_trainer import CONFIG, Actor  # noqa: E402

ckpt_path = sys.argv[1]
n_games = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 50
params = {}
for a in sys.argv[2:]:
    if "=" in a:
        k, v = a.lstrip("-").replace("param ", "").split("=", 1)
        params[k] = float(v)

ck = torch.load(ckpt_path, map_location="cpu")
actors = {}
for role in ("human", "saboteur", "captain"):
    act = Actor(CONFIG["local_obs_dim"], CONFIG["action_branches"], CONFIG["hidden_dim"], CONFIG["num_layers"])
    act.load_state_dict(ck[f"actor_{role}"])
    act.eval()
    actors[role] = act

shots = []
orig_execute = ts.TheThingWorld._execute


def logged_execute(self, shooter, t):
    others = self._other_slots(shooter)
    obs = self._collect_obs(shooter)
    slot = others.index(t)
    shots.append({
        "target_role": ["human", "saboteur", "captain"][self.role[t.idx]],
        "witness_flag": float(obs[40 + 4 * slot + 3]),
        "any_flag": float(max(obs[40 + 4 * k + 3] for k in range(len(others)))),
        "time": round(self.timer, 1),
        "dist": round(ts.dist(shooter.pos, t.pos), 1),
        "target_state": "repair" if (t.interacting and not t.sabotaging) else ("sabotage" if t.sabotaging else t.state),
    })
    return orig_execute(self, shooter, t)


ts.TheThingWorld._execute = logged_execute

log = []
w = ts.TheThingWorld(seed=123, stats_sink={}, game_log=log)
w.env_params = dict(params)
w.load_env_params()
w._reset_state()
w.force_reset()
out = w.first_exchange()
L = CONFIG["local_obs_dim"]
while len(log) < n_games:
    acts = {}
    for aid, obs, r, done in out:
        if done:
            continue
        role = "captain" if obs[1] > 0.5 else ("saboteur" if obs[0] > 0.5 else "human")
        with torch.no_grad():
            a, _ = actors[role].get_action(torch.FloatTensor(obs[:L]).unsqueeze(0))
        acts[aid] = tuple(int(x) for x in a.squeeze(0).tolist())
    out = w.advance(acts)

print("games", len(log), "human win", np.mean([g["human_win"] for g in log]))
print("shots", len(shots), "target", collections.Counter(s["target_role"] for s in shots))
print("목격 플래그가 켜진 대상을 쐈나:", collections.Counter(s["witness_flag"] for s in shots))
print("사격 순간 함장 관측에 목격 플래그가 하나라도 있었나:", collections.Counter(s["any_flag"] for s in shots))
print("대상 상태:", collections.Counter(s["target_state"] for s in shots))
print("사격 시각(초) 분포:", np.percentile([s["time"] for s in shots], [10, 50, 90]) if shots else None)
print("거리:", np.percentile([s["dist"] for s in shots], [10, 50, 90]) if shots else None)
