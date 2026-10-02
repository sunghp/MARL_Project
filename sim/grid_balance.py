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
    # 4차: v3_chosen 기준 인간팀 학습용 밸런스
    "v4": {
        "ship_speed_by_health": [0.0, 1.0],
        "spawn_spread": [0.0, 1.0],
        "repair_range": [2.0, 5.0],
        "repeat_damage_decay": [1.0, 0.95],
        "sabotage_damage": [10.0, 7.0],
    },
    # 4차-2: 자동 수리
    "v4b": {
        "auto_repair": [0.0, 1.0],
        "spawn_spread": [0.0, 1.0],
        "repair_range": [2.0, 5.0],
        "sabotage_damage": [10.0, 7.0],
        "ship_speed_by_health": [1.0],
    },
    # 4차-3: v4 규칙에서 사보타주 쪽 신호 회복 (무작위 사보타주도 가끔 이기게)
    "v4c": {
        "total_distance": [1000.0, 1300.0, 1600.0],
        "sabotage_damage": [7.0, 10.0],
        "alert_threshold": [60.0, 70.0],
    },
    # 4차-4: 사보타주 쪽 행동열 단축 (자동 부수기)
    "v4d": {
        "auto_sabotage": [0.0, 1.0],
        "sabotage_damage": [7.0, 8.5],
        "total_distance": [1000.0, 1300.0],
    },
}
BASE = {"v2": "v1_chosen", "v3": "v2_chosen", "v4": "v3_chosen", "v4b": "v3_chosen", "v4c": "v4_rules", "v4d": "v4_rules"}
MATCHUPS = [("camper_vs_bot", 2, "camper"), ("bot_vs_bot", 3, "random"), ("rand_vs_rand", 0, "random"),
            # 무작위 인간팀 vs 방 붙기 사보타주: 학습 전 인간팀도 가끔 이겨야 학습 신호가 생긴다
            ("rand_vs_camper", 0, "camper"),
            # 무작위 사보타주 vs 규칙봇 인간팀: 학습 전 사보타주도 가끔 이겨야 한다
            ("rand_sab_vs_bot", 2, "random"),
            ("rand_hum_vs_bot", 1, "random")]


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
