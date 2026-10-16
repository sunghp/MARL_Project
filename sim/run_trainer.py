"""
run_trainer.py - 가상 환경(thething_sim)에서 mappo_trainer.py를 그대로 실행

  python sim/run_trainer.py --out runs_dir/expA --seed 0 --timesteps 2000000
  (옵션) --eval-interval N --eval-episodes K : 평가 주기/길이 덮어쓰기

out 디렉터리에 TensorBoard 로그(runs/mappo), 체크포인트, games.jsonl(게임별 ground truth)이 생긴다.
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "fake_mlagents"))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timesteps", type=int, default=None)
ap.add_argument("--eval-interval", type=int, default=None)
ap.add_argument("--eval-episodes", type=int, default=None)
ap.add_argument("--param", action="append", default=[],
                help="밸런스 파라미터 덮어쓰기 key=value (environment_parameters), 여러 번 지정 가능")
ap.add_argument("--entropy-coef", type=float, default=None, help="PPO 엔트로피 계수 덮어쓰기 (기본 CONFIG 0.01)")
ap.add_argument("--entropy-role", action="append", default=[],
                help="역할별 엔트로피 계수 role=value (예: saboteur=0.003), 여러 번 지정 가능")
ap.add_argument("--lr-actor", type=float, default=None, help="Actor 학습률 덮어쓰기 (기본 CONFIG 1e-4)")
ap.add_argument("--action-mask", action="store_true", help="역할별로 효과 없는 상호작용을 막음")
ap.add_argument("--min-update-samples", type=int, default=None, help="이 샘플 수가 모일 때까지 여러 판을 묶어 업데이트")
ap.add_argument("--actor-freeze-steps", type=int, default=None, help="이 스텝까지 Actor 고정, Critic만 학습")
ap.add_argument("--reset-steps", action="store_true", help="--load 후 스텝/에피소드 수를 0으로 (모방학습 체크포인트에서 시작)")
ap.add_argument("--load", default=None, help="이어서 학습할 체크포인트(.pt) 경로 (total_steps, 옵티마이저 상태 포함 복원)")
ap.add_argument("--curriculum", default=None,
                help="규칙봇 커리큘럼 bot_team:start:end:decay_steps (예: 1:1.0:0.3:500000)")
args = ap.parse_args()

os.makedirs(os.path.join(args.out, "config"), exist_ok=True)
with open(os.path.join(ROOT, "config", "TheThing.yaml"), encoding="utf-8") as f:
    yaml_text = f.read()
if args.param:
    import yaml
    cfg_yaml = yaml.safe_load(yaml_text)
    for kv in args.param:
        k, v = kv.split("=", 1)
        cfg_yaml.setdefault("environment_parameters", {})[k] = float(v)
    yaml_text = yaml.safe_dump(cfg_yaml, allow_unicode=True, sort_keys=False)
with open(os.path.join(args.out, "config", "TheThing.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_text)
os.chdir(args.out)
os.environ["SIM_SEED"] = str(args.seed)
os.environ["SIM_GAME_LOG"] = "games.jsonl"
if os.path.exists("games.jsonl"):
    os.remove("games.jsonl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.set_num_threads(1)

import mappo_trainer  # noqa: E402

cfg = dict(mappo_trainer.CONFIG)
if args.timesteps:
    cfg["total_timesteps"] = args.timesteps
if args.eval_interval:
    cfg["eval_interval"] = args.eval_interval
if args.eval_episodes:
    cfg["eval_episodes"] = args.eval_episodes
if args.entropy_coef is not None:
    cfg["entropy_coef"] = args.entropy_coef
if args.action_mask:
    cfg["action_mask"] = True
if args.min_update_samples is not None:
    cfg["min_update_samples"] = args.min_update_samples
if args.actor_freeze_steps is not None:
    cfg["actor_freeze_steps"] = args.actor_freeze_steps
if args.lr_actor is not None:
    cfg["lr_actor"] = args.lr_actor
if args.entropy_role:
    cfg["entropy_coef_by_role"] = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.entropy_role)}
if args.curriculum:
    team, start, end, decay = args.curriculum.split(":")
    cfg["bot_curriculum"] = {"bot_team": int(team), "start": float(start), "end": float(end),
                             "decay_steps": int(decay)}

trainer = mappo_trainer.MAPPOTrainer(cfg)
if args.load:
    trainer.load(os.path.abspath(args.load) if not os.path.isabs(args.load) else args.load)
    if args.reset_steps:
        trainer.total_steps = 0
        trainer.episode_count = 0
        trainer.next_eval_step = cfg["eval_interval"]
trainer.train()
