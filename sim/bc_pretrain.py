"""
bc_pretrain.py - 규칙봇 사보타주의 시연으로 사보타주 Actor를 미리 학습 (모방학습, behavior cloning)

  python sim/bc_pretrain.py --games 3000 --out checkpoints_bc/bc_saboteur.pt [--param sab_obs=1 ...]

1. 전원 규칙봇(frozen_team 3)으로 게임을 돌리며, 사보타주의 결정 시점마다
   - 관측: 학습 사보타주가 받을 관측과 같은 것 (로컬 64차원)
   - 정답 행동: 규칙봇이 실제로 하려는 것
     방 = 지금 목표 방 (없으면 대기 8), 상호작용 = 부수려는 중이고 시야에 인간팀이 없으면 부수기(1), 아니면 0
   을 모은다.
2. 교차 엔트로피로 사보타주 Actor(행동 막기 적용)를 학습한다.
3. 트레이너가 --load로 이어 받을 수 있는 체크포인트를 저장한다 (스텝 0, Critic 없음).
   PPO는 --reset-steps --actor-freeze-steps로 Critic을 먼저 예열하고 시작한다.

규칙봇은 인간팀 위치를 모두 알고 고르지만 학습 사보타주는 자기 시야만 보므로, 완전히 따라 할 수는 없다.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fake_mlagents"))
sys.path.insert(0, ROOT)

from bot_ladder import yaml_params  # noqa: E402
from mappo_trainer import CONFIG, make_actor  # noqa: E402
from thething_sim import SABOTEUR, TheThingWorld  # noqa: E402

L = CONFIG["local_obs_dim"]


def collect(args):
    params, n, seed = args
    log = []
    w = TheThingWorld(seed=seed, stats_sink={}, game_log=log)
    w.env_params = dict(params, eval_mode=1.0, frozen_team=3.0, bot_team=0.0)
    w.load_env_params()
    w._reset_state()
    w.force_reset()
    out = w.first_exchange()
    X, Y = [], []
    while len(log) < n:
        by_id = {c.agent.episode_id: c for c in w.chars}
        for aid, obs, r, done in out:
            c = by_id.get(aid)
            if done or c is None or w.role[c.idx] != SABOTEUR or not c.active:
                continue
            room = c.room.idx if c.room is not None else 8
            sab = 1 if (c.bot_wants_sab and c.room is not None and not w._crew_in_sight(c)) else 0
            X.append(obs[:L])
            Y.append((room, sab))
        out = w.advance({})
    return np.asarray(X, np.float32), np.asarray(Y, np.int64), [g["human_win"] for g in log]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param", action="append", default=[])
    args = ap.parse_args()
    params = yaml_params()
    for kv in args.param:
        k, v = kv.split("=", 1)
        params[k] = float(v)

    workers = 4
    per = args.games // workers
    with Pool(workers) as pool:
        parts = pool.map(collect, [(params, per, args.seed * 100 + i) for i in range(workers)])
    X = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    hw = [h for p in parts for h in p[2]]
    print(f"[시연] {len(hw)}판, 규칙봇 사보타주 승률 {1 - np.mean(hw):.3f}, 샘플 {len(X)}, "
          f"부수기 라벨 비율 {Y[:, 1].mean():.3f}, 대기 라벨 비율 {(Y[:, 0] == 8).mean():.3f}")

    torch.manual_seed(args.seed)
    actor = make_actor("saboteur", action_mask=True)
    perm = np.random.default_rng(args.seed).permutation(len(X))
    n_val = len(X) // 10
    val, tr = perm[:n_val], perm[n_val:]
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    hist = []
    for ep in range(args.epochs):
        actor.train()
        np.random.shuffle(tr)
        for i in range(0, len(tr), 1024):
            b = tr[i:i + 1024]
            lr_, li_ = actor(Xt[b])
            loss = ce(lr_, Yt[b, 0]) + ce(li_, Yt[b, 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
        actor.eval()
        with torch.no_grad():
            lr_, li_ = actor(Xt[val])
            vloss = (ce(lr_, Yt[val, 0]) + ce(li_, Yt[val, 1])).item()
            acc_r = (lr_.argmax(-1) == Yt[val, 0]).float().mean().item()
            acc_i = (li_.argmax(-1) == Yt[val, 1]).float().mean().item()
        hist.append({"epoch": ep + 1, "val_loss": round(vloss, 4), "room_acc": round(acc_r, 3), "sab_acc": round(acc_i, 3)})
        print(f"[BC] epoch {ep + 1}: val loss {vloss:.3f}, 방 정확도 {acc_r:.3f}, 부수기 정확도 {acc_i:.3f}")

    # 트레이너 체크포인트 형식 (다른 역할은 무작위 초기값)
    ck = {"episode": 0, "total_steps": 0, "action_mask": True, "bc": {"games": len(hw), "samples": int(len(X)),
          "teacher_sab_win": round(1 - float(np.mean(hw)), 3), "params": params, "history": hist}}
    for role in ("human", "captain"):
        ck[f"actor_{role}"] = make_actor(role, action_mask=True).state_dict()
    ck["actor_saboteur"] = actor.state_dict()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    print("[저장]", args.out, json.dumps(hist[-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
