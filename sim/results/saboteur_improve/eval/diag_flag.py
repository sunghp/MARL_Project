import sys, numpy as np
sys.path.insert(0,'sim')
import thething_sim as ts
from eval_saboteur import play, load_actors
from bot_ladder import yaml_params
W=ts.TheThingWorld
flag={}; out=[]
o_w, o_exec, o_end = W._witnesses, W._execute, W._notify_game_end
def wit(self,sab,record):
    n=o_w(self,sab,record)
    if record and n>0 and (self.game_index,sab.idx) not in flag:
        flag[(self.game_index,sab.idx)]=dict(t=self.timer, interacting=sab.interacting, dead_t=None)
    return n
def exe(self,sh,t):
    k=(self.game_index,t.idx)
    if k in flag: flag[k]['dead_t']=self.timer
    else: out.append('unflagged')
    return o_exec(self,sh,t)
W._witnesses, W._execute = wit, exe
sab=load_actors('models/saboteur_vs_bot_8M.pt',['saboteur'])['saboteur']
log=play(yaml_params(),sab,None,200,99)
f=list(flag.values())
dead=[x for x in f if x['dead_t'] is not None]
print("목격된 사보타주(판당)",round(len(f)/200,2),"그중 처형",round(len(dead)/len(f),3),"목격→처형 시간 중앙값",round(float(np.median([x['dead_t']-x['t'] for x in dead])),1),"근거 없이 처형",len(out))
