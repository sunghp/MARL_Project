"""
grid_balance.py - V1(채택값) 주변 밸런스 격자 탐색 (병렬)

  python sim/grid_balance.py [게임 수] [v2|v3] > grid.jsonl
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from tune_balance import PRESETS, play, summary  # noqa: E402

GRIDS = {
    # 2차: V1 주변
    "v2": {
        "vision_range": [10.0, 12.0],
        "alert_threshold": [60.0, 70.0],
        "share_witness": [0.0, 1.0],
        "ship_stop_threshold": [50.0, 70.0],
    },
    # 3차: v2_chosen 기준 함장 사격 규칙
    "v3": {
        "shoot_requires_evidence": [0.0, 1.0],
        "shoot_range": [-1.0, 25.0, 0.0],
        "captain_bullets": [1.0, 2.0],
        "alert_threshold": [60.0, 70.0],
    },
}
BASE = {"v2": "v1_chosen", "v3": "v2_chosen"}
MATCHUPS = [("camper_vs_bot", 2, "camper"), ("bot_vs_bot", 3, "random"), ("rand_vs_rand", 0, "random")]


def one(args):
    params, n = args
    return {"params": params, **{m: summary(play(n, params, fr, pol)) for m, fr, pol in MATCHUPS}}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    name = sys.argv[2] if len(sys.argv) > 2 else "v3"
    grid = GRIDS[name]
    keys = list(grid)
    jobs = [(dict(PRESETS[BASE[name]], **dict(zip(keys, vals))), n)
            for vals in itertools.product(*grid.values())]
    with Pool(4) as pool:
        for r in pool.imap(one, jobs):
            print(json.dumps(r, ensure_ascii=False), flush=True)
