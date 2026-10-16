import sys, collections, numpy as np
sys.path.insert(0,'sim')
import thething_sim as ts
from eval_saboteur import play, load_actors
from bot_ladder import yaml_params, play as bplay
W=ts.TheThingWorld
hits=collections.defaultdict(list)
o=W._complete
def comp(self,c):
    if c.sabotaging: hits[self.game_index].append(c.room.idx)
    return o(self,c)
W._complete=comp
name, ck, v2 = sys.argv[1], sys.argv[2], sys.argv[3]=="1"
p=dict(yaml_params(), sab_obs=1.0 if v2 else 0.0)
if ck=="rulebot": log=bplay(p,"rulebot",150,5)
else: log=play(p, load_actors(ck,['saboteur'])['saboteur'], None, 150, 5)
fr=[collections.Counter(h).most_common(1)[0][1]/len(h) for h in hits.values() if len(h)>=3]
nr=[len(set(h)) for h in hits.values() if h]
print(name, "승률", round(1-np.mean([g['human_win'] for g in log]),3), "가장 많이 부순 방 비율", round(np.mean(fr),2), "판당 부순 방 수", round(np.mean(nr),1))
