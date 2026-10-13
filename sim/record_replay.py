"""
record_replay.py - 시뮬레이터에서 게임을 돌려 리플레이(JSON)로 저장

  python sim/record_replay.py --checkpoint models/sim_balance_v4_4M_seed2.pt --games 100 --keep 4 \\
      --out sim/results/replay/replay.json --html sim/results/replay/replay.html

- 세 조합으로 게임을 돌린다: 전원 학습 정책("trained"), 사보타주만 학습("sab_only"), 전원 무작위("random"). 규칙은 config/TheThing.yaml과 같다.
- 결정 시점(0.4초)마다 프레임을 남긴다: 캐릭터 위치/상태, 방 안정도/알림/사용 중, 함선 진행도.
- 사건: 부수기 완료, 수리 완료, 끊기, 목격(첫 목격), 함장 신고 수신, 사격, 게임 종료.
- 전체 게임의 요약 통계와 함께, 종료 사유가 골고루 섞이도록 고른 몇 판의 프레임을 저장한다.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fake_mlagents"))
sys.path.insert(0, ROOT)

import thething_sim as ts  # noqa: E402
from mappo_trainer import CONFIG, Actor  # noqa: E402

ROLE_NAME = {ts.HUMAN: "crew", ts.SABOTEUR: "saboteur", ts.CAPTAIN: "captain"}


def load_params():
    with open(os.path.join(ROOT, "config", "TheThing.yaml"), encoding="utf-8") as f:
        return {k: float(v) for k, v in yaml.safe_load(f)["environment_parameters"].items()}


class Recorder:
    active = None                 # 패치된 메서드가 기록을 보낼 현재 Recorder
    _patched = False

    def __init__(self, world):
        self.w = world
        self.games = {}           # game_index -> dict
        Recorder.active = self
        if not Recorder._patched:
            Recorder._patched = True
            self._patch()

    def game(self):
        gi = self.w.game_index
        if gi not in self.games:
            w = self.w
            self.games[gi] = {
                "roles": [ROLE_NAME[w.role[c.idx]] for c in w.chars],
                "frames": [], "events": [], "end": None,
            }
        return self.games[gi]

    def ev(self, kind, **kw):
        self.game()["events"].append(dict(t=round(self.w.timer, 2), kind=kind, **kw))

    @staticmethod
    def _patch():
        W = ts.TheThingWorld
        orig_complete, orig_try_start, orig_execute = W._complete, W._try_start, W._execute
        orig_hw, orig_sw = W.human_win, W.sabotage_win

        def complete(self, c):
            rec = Recorder.active
            room, sab = c.room, c.sabotaging
            before = room.health
            orig_complete(self, c)
            if sab:
                rec.ev("sabotage", who=c.idx, room=room.idx, health=round(room.health, 1), hidden=not c.seen)
            elif room.health > before:
                rec.ev("repair", who=c.idx, room=room.idx, health=round(room.health, 1))

        def try_start(self, c, sabotage):
            rec = Recorder.active
            victim = c.room.used_by if c.room is not None else None
            n0 = self.sab_interrupted
            ok = orig_try_start(self, c, sabotage)
            if self.sab_interrupted > n0 and victim is not None:
                rec.ev("interrupt", who=c.idx, victim=victim.idx, room=c.room.idx)
            return ok

        def execute(self, shooter, t):
            rec = Recorder.active
            hit = self.role[t.idx] == ts.SABOTEUR
            rec.ev("shot", who=shooter.idx, target=t.idx, hit=hit)
            return orig_execute(self, shooter, t)

        def end(winner, orig):
            def f(self, reason):
                rec = Recorder.active
                if self.game_over:
                    return
                rec.snapshot()                      # 리셋 전 마지막 상태
                g = rec.game()
                g["end"] = {"t": round(self.timer, 2), "winner": winner, "reason": reason}
                return orig(self, reason)
            return f

        W._complete, W._try_start, W._execute = complete, try_start, execute
        W.human_win, W.sabotage_win = end("crew", orig_hw), end("saboteur", orig_sw)

    def snapshot(self):
        w = self.w
        g = self.game()
        chars = []
        for c in w.chars:
            state = 0
            if c.interacting:
                state = 1 if c.sabotaging else 2
            chars.append([round(float(c.pos[0]), 1), round(float(c.pos[1]), 1), 1 if c.active else 0, state])
        rooms = [[round(r.health, 1), 1 if r.alerted else 0, -1 if r.used_by is None else r.used_by.idx]
                 for r in w.rooms]
        g["frames"].append({
            "t": round(w.timer, 2),
            "p": round(w.distance / w.p("total_distance"), 4),
            "c": chars, "r": rooms,
        })
        # 새로 생긴 목격 기록 → 사건
        prev = g.setdefault("_seen", set())
        for obs, sus in w.witnessed.items():
            for s in sus:
                if (obs, s) not in prev:
                    prev.add((obs, s))
                    kind = "report" if (w.captain is not None and obs == w.captain.idx) else "witness"
                    g["events"].append({"t": round(w.timer, 2), "kind": kind, "who": obs, "suspect": s})


def run(policy, actors, n_games, seed, params):
    """트레이너와 같은 루프로 돌린다.
    - 게임마다 처음 받은 관측으로 역할을 정하고, 등록된 에이전트가 모두 종료되면 env.reset()
    - 리셋을 빼면 결정 타이밍과 저장된 행동이 학습 때와 달라져 통계가 어긋난다."""
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

    pc = EnvironmentParametersChannel()
    for k, v in params.items():
        pc.set_float_parameter(k, v)
    os.environ["SIM_SEED"] = str(seed)
    env = UnityEnvironment(side_channels=[pc])
    # 시뮬레이터 첫 판은 코드 기본값(옛 규칙)으로 시작하므로, 설정을 넣고 다시 리셋해 첫 판부터 같은 규칙으로
    env.world.env_params = dict(params)
    env.world._reset_state()
    rec = Recorder(env.world)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.reset()
    start_game = env.world.game_index
    rec.snapshot()
    L = CONFIG["local_obs_dim"]
    roles, term = {}, set()
    ident = lambda o: "captain" if o[1] > 0.5 else ("saboteur" if o[0] > 0.5 else "human")
    while len([gi for gi, g in rec.games.items() if gi >= start_game and g["end"]]) < n_games:
        d, t = env.get_steps("TheThing")
        for aid in t.agent_id:
            if int(aid) in roles:
                term.add(int(aid))
        if roles and set(roles).issubset(term):
            roles.clear()
            term.clear()
            env.reset()
            rec.snapshot()
            continue
        if len(d) == 0:
            env.step()
            rec.snapshot()
            continue
        acts = np.zeros((len(d), 2), dtype=np.int32)
        for i, aid in enumerate(d.agent_id):
            aid = int(aid)
            obs = d[aid].obs[0]
            roles.setdefault(aid, ident(obs))
            if policy == "random" or (policy == "sab_only" and roles[aid] != "saboteur"):
                acts[i] = rng.integers(0, 9, size=2)
            else:
                with torch.no_grad():
                    a, _ = actors[roles[aid]].get_action(torch.FloatTensor(obs[:L]).unsqueeze(0))
                acts[i] = a.squeeze(0).numpy()
        env.set_actions("TheThing", ActionTuple(discrete=acts))
        env.step()
        rec.snapshot()
    games = [g for gi, g in sorted(rec.games.items()) if gi >= start_game and g["end"]]
    for g in games:
        g.pop("_seen", None)
    return games[:n_games]


def summarize(games):
    n = len(games)
    reasons = collections.Counter()
    for g in games:
        r = g["end"]["reason"]
        if "파괴" in r:
            r = "방 완전 파괴"
        reasons[r] += 1
    ev = lambda k: round(sum(1 for g in games for e in g["events"] if e["kind"] == k) / n, 2)
    shots = [e for g in games for e in g["events"] if e["kind"] == "shot"]
    return {
        "games": n,
        "crew_win": round(sum(g["end"]["winner"] == "crew" for g in games) / n, 3),
        "duration": round(float(np.mean([g["end"]["t"] for g in games])), 1),
        "sabotage": ev("sabotage"), "repair": ev("repair"), "interrupt": ev("interrupt"),
        "witness": ev("witness"), "shots": ev("shot"),
        "shot_hit": round(sum(e["hit"] for e in shots) / len(shots), 3) if shots else None,
        "reasons": dict(reasons.most_common()),
    }


def pick(games, k):
    """종료 사유가 섞이도록 k판 선택"""
    by = collections.defaultdict(list)
    for g in games:
        r = g["end"]["reason"]
        by["destroy" if "파괴" in r else ("exec" if "제거" in r else ("parity" if "동수" in r else "arrive"))].append(g)
    out = []
    order = ["destroy", "exec", "arrive", "parity"]
    while len(out) < k and any(by.values()):
        for key in order:
            if by[key] and len(out) < k:
                out.append(by[key].pop(0))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=40, help="통계용 게임 수 (정책별)")
    ap.add_argument("--keep", type=int, default=4, help="리플레이로 저장할 판 수 (정책별)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="리플레이 JSON 경로")
    ap.add_argument("--html", default="", help="리플레이 뷰어 HTML도 함께 저장 (데이터 내장)")
    args = ap.parse_args()

    params = load_params()
    ck = torch.load(args.checkpoint, map_location="cpu")
    actors = {}
    for role in ("human", "saboteur", "captain"):
        a = Actor(CONFIG["local_obs_dim"], CONFIG["action_branches"], CONFIG["hidden_dim"], CONFIG["num_layers"])
        a.load_state_dict(ck[f"actor_{role}"])
        a.eval()
        actors[role] = a

    result = {
        "map": {"rooms": [{"name": n, "x": p[0], "z": p[1]} for n, p in ts.ROOMS], "cafe": list(ts.CAFE)},
        "params": params,
        "checkpoint": {"file": os.path.basename(args.checkpoint), "total_steps": int(ck.get("total_steps", 0))},
        "decision_dt": ts.DT * ts.DECISION_PERIOD,
        "policies": {},
    }
    # trained: 모두 학습 정책 / sab_only: 사보타주만 학습, 인간팀은 무작위 / random: 모두 무작위
    for policy in ("trained", "sab_only", "random"):
        games = run(policy, actors, args.games, args.seed, params)
        result["policies"][policy] = {"summary": summarize(games), "replays": pick(games, args.keep)}
        print(policy, json.dumps(result["policies"][policy]["summary"], ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print(args.out, os.path.getsize(args.out), "bytes")
    if args.html:
        with open(os.path.join(HERE, "replay_viewer_template.html"), encoding="utf-8") as f:
            page = f.read()
        data = json.dumps(result, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(page.replace("/*__DATA__*/null", data))
        print(args.html, os.path.getsize(args.html), "bytes")
