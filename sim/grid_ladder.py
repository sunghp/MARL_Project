"""
grid_ladder.py - 강화 봇(bot_level 1) 기준 밸런스 격자. 조합마다 사보타주 전략 사다리를 잰다.

  python sim/grid_ladder.py [게임 수] [격자 이름] > out.jsonl
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot_ladder import POLICIES, play, summary, yaml_params  # noqa: E402

GRIDS = {
    # 회피형 사보타주 봇 반영 후. g1(일부)에서 피해 7은 모든 전략이 10% 이하라 제외
    "g2": {
        "sabotage_damage": [8.0, 9.0, 10.0],
        "vision_range": [10.0, 12.0],
        "captain_bullets": [1.0, 2.0],
        "alert_threshold": [60.0, 70.0],
        "shoot_range": [15.0],
    },
    "g1": {
        "vision_range": [10.0, 12.0],
        "captain_bullets": [1.0, 2.0],
        "sabotage_damage": [7.0, 9.0],
        "alert_threshold": [60.0, 70.0],
        "shoot_range": [15.0, 25.0],
    },
}


def job(a):
    i, params, pol, n = a
    return i, pol, summary(play(params, pol, n))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    grid = GRIDS[sys.argv[2] if len(sys.argv) > 2 else "g1"]
    base = dict(yaml_params(), bot_level=1.0)
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    jobs = [(i, dict(base, **c), p, n) for i, c in enumerate(combos) for p in POLICIES]
    res = {}
    with Pool(4) as pool:
        for i, pol, s in pool.imap_unordered(job, jobs):
            res.setdefault(i, {})[pol] = s
            if len(res[i]) == len(POLICIES):
                print(json.dumps({"params": combos[i], **{p: res[i][p] for p in POLICIES}}, ensure_ascii=False), flush=True)
