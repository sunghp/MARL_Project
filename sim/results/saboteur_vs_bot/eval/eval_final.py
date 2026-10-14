import sys, json
sys.path.insert(0, 'sim')
from multiprocessing import Pool
from eval_saboteur import _job
from bot_ladder import yaml_params
SP = sys.argv[1]
cks = {"C8_s0": "svb_C_s0_c/checkpoints/mappo_ep7550.pt", "C8_s1": "svb_C_s1_c/checkpoints/mappo_ep7550.pt"}
if __name__ == "__main__":
    base = yaml_params()
    jobs = [(k, base, f"{SP}/{p}", None, 500, 777) for k, p in cks.items()]
    jobs += [("B8_s1", base, f"{SP}/svb_B_s1_c/checkpoints/mappo_ep7400.pt", None, 500, 777),
             ("B8_s0", base, f"{SP}/svb_B_s0_c/checkpoints/mappo_ep7550.pt", None, 500, 777)]
    with Pool(4) as pool:
        for name, s in pool.imap(_job, jobs):
            print(name, json.dumps(s, ensure_ascii=False), flush=True)
