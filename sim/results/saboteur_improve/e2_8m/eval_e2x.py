"""6개 시드 최종 체크포인트를 500판(시드 777)으로 평가. 사용: python eval_e2x.py SP"""
import sys, json, os
sys.path.insert(0, 'sim')
from multiprocessing import Pool
from eval_saboteur import _job
from bot_ladder import yaml_params
SP = sys.argv[1]
def last(d):
    n = max(int(f[8:-3]) for f in os.listdir(f"{SP}/{d}/checkpoints")); return f"{SP}/{d}/checkpoints/mappo_ep{n}.pt"
if __name__ == "__main__":
    v2 = dict(yaml_params(), sab_obs=1.0)
    dirs = ["e2x_s0_c", "e2x_s1_c"] + [f"e2x_s{s}" for s in range(2, 6)]
    with Pool(4) as pool:
        for name, s in pool.imap(_job, [(d, v2, last(d), None, 500, 777) for d in dirs]):
            print(name, json.dumps(s, ensure_ascii=False), flush=True)
