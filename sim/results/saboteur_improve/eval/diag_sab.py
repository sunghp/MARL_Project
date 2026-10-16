import sys, collections, numpy as np
sys.path.insert(0,'sim')
import thething_sim as ts
from eval_saboteur import play, load_actors
from bot_ladder import yaml_params
W=ts.TheThingWorld
st=collections.Counter(); deaths=[]; starts=[]
o_try, o_exec = W._try_start, W._execute
def try_start(self,c,sab):
    ok=o_try(self,c,sab)
    if ok and sab and self.role[c.idx]==ts.SABOTEUR:
        vis=self.p("vision_range")
        crew=[o for o in self.alive if o.active and self.role[o.idx]!=ts.SABOTEUR]
        near=min((ts.dist(o.pos,c.pos) for o in crew), default=999)
        cap=self.captain
        capd=ts.dist(cap.pos,c.pos) if cap is not None and cap.active else 999
        other=[s for s in self.saboteurs if s is not c]
        same=bool(other) and other[0].room is c.room
        starts.append(dict(insight=near<=vis, near=near, capd=capd, alerted=c.room.alerted, health=c.room.health, same_room=same, t=self.timer))
    return ok
def execute(self,sh,t):
    if self.role[t.idx]==ts.SABOTEUR:
        deaths.append(dict(t=self.timer, interacting=t.interacting))
    return o_exec(self,sh,t)
W._try_start, W._execute = try_start, execute
p=yaml_params()
sab=load_actors('models/saboteur_vs_bot_8M.pt',['saboteur'])['saboteur']
log=play(p,sab,None,200,99)
n=len(starts)
print("게임",len(log),"사보 승",round(1-np.mean([g['human_win'] for g in log]),3))
print("부수기 시작",n,"/판",round(n/len(log),1))
print(" 시작 순간 인간팀이 시야(10) 안:",round(np.mean([s['insight'] for s in starts]),3))
print(" 시작 순간 가장 가까운 인간팀 거리 중앙값:",round(float(np.median([s['near'] for s in starts])),1))
print(" 시작 순간 함장 25 이내:",round(np.mean([s['capd']<=25 for s in starts]),3))
print(" 이미 알림 뜬 방(<=60):",round(np.mean([s['alerted'] for s in starts]),3))
print(" 동료와 같은 방 목표:",round(np.mean([s['same_room'] for s in starts]),3))
print("처형",len(deaths),"/판",round(len(deaths)/len(log),2),"첫 처형 시각 중앙값",round(float(np.median([d['t'] for d in deaths])),1),"부수는 중 처형 비율",round(np.mean([d['interacting'] for d in deaths]),2))
r=collections.Counter(("파괴" if "파괴" in g['reason'] else g['reason']) for g in log); print(r)
print("완료 부수기/판",round(np.mean([g['sab_total'] for g in log]),1),"들킨 완료/판",round(np.mean([g['sab_total']-g['sab_hidden'] for g in log]),2),"끊김/판",round(np.mean([g['sab_interrupted'] for g in log]),2))
