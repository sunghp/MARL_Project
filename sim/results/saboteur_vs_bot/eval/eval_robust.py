import sys, json
sys.path.insert(0, 'sim')
from multiprocessing import Pool
from eval_saboteur import _job, VARIANTS
from bot_ladder import yaml_params
SP, tag, ck = sys.argv[1], sys.argv[2], sys.argv[3]
crew = "models/sim_balance_v4_4M_seed2.pt"
if __name__ == "__main__":
    base = yaml_params()
    jobs = [(f"{tag}|{v}", dict(base, **VARIANTS[v]), ck, None, 300, 4242) for v in VARIANTS]
    if not ck in ("rulebot", "camper", "evasive"):
        jobs.append((f"{tag}|learned_crew", base, ck, crew, 300, 4242))
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 2) as pool:
        for name, s in pool.imap(_job, jobs):
            print(name, json.dumps(s, ensure_ascii=False), flush=True)
