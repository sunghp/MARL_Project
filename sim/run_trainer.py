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

trainer = mappo_trainer.MAPPOTrainer(cfg)
trainer.train()
