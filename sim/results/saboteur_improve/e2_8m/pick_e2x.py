import sys, os, json, numpy as np
sys.path.insert(0, 'sim')
from multiprocessing import Pool
from analyze import load_games, rolling
from eval_saboteur import _job
from bot_ladder import yaml_params
SP = sys.argv[1]
RUNS = {"s0": ("e2x_m_s0", ["imp_E2_s0", "e2x_s0_c"]), "s1": ("e2x_m_s1", ["imp_E2_s1", "e2x_s1_c"])}
for s in range(2, 6):
    RUNS[f"s{s}"] = (f"e2x_s{s}", [f"e2x_s{s}"])
def ckpts(dirs):
    out = {}
    for d in dirs:
        for f in os.listdir(f"{SP}/{d}/checkpoints"):
            out[int(f[8:-3])] = f"{SP}/{d}/checkpoints/{f}"
    return out
jobs = []
for name, (gdir, cdirs) in RUNS.items():
    g = load_games(f"{SP}/{gdir}")
    win = np.array([np.nan if (x["eval_mode"] or not x.get("bot_team")) else 1 - x["human_win"] for x in g])
    idx = np.arange(len(g))
    tr = ~np.isnan(win)
    r = rolling(win[tr], 300)
    ti = idx[tr]
    ok = ti >= int(len(g) * 0.55)          # 대략 4.5M 이후
    centre = ti[ok][int(np.argmax(r[ok]))] - 150
    ck = ckpts(cdirs)
    eps = sorted(ck)
    near = sorted(eps, key=lambda e: abs(e - centre))[:3]
    for e in near:
        jobs.append((f"{name}_ep{e}", ck[e]))
if __name__ == "__main__":
    v2 = dict(yaml_params(), sab_obs=1.0)
    with Pool(4) as pool:
        for n, s in pool.imap(_job, [(n, v2, p, None, 300, 777) for n, p in jobs]):
            print(n, json.dumps(s, ensure_ascii=False), flush=True)
