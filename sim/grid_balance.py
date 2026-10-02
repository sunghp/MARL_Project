"""
grid_balance.py - V1(채택값) 주변 밸런스 격자 탐색 (병렬)

  python sim/grid_balance.py [게임 수] > grid.jsonl
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from tune_balance import PRESETS, play, summary  # noqa: E402

GRID = {
    "vision_range": [10.0, 12.0],
    "alert_threshold": [60.0, 70.0],
    "share_witness": [0.0, 1.0],
    "ship_stop_threshold": [50.0, 70.0],
}
MATCHUPS = [("camper_vs_bot", 2, "camper"), ("bot_vs_bot", 3, "random")]


def one(args):
    params, n = args
    return {"params": params, **{m: summary(play(n, params, fr, pol)) for m, fr, pol in MATCHUPS}}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    keys = list(GRID)
    jobs = [(dict(PRESETS["v1_chosen"], **dict(zip(keys, vals))), n)
            for vals in itertools.product(*GRID.values())]
    with Pool(4) as pool:
        for r in pool.imap(one, jobs):
            print(json.dumps(r, ensure_ascii=False), flush=True)
