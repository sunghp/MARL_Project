"""
thething_sim.py - Unity "The Thing" 게임의 파이썬 포팅 (가상 환경)

Unity 빌드 없이 mappo_trainer.py를 그대로 돌려보기 위한 시뮬레이터.
C# 스크립트(GameManager / SystemHealth / InteractionPoint / NPCController / NPCAgent /
CaptainGun / RoleManager)의 로직과 ML-Agents(com.unity.ml-agents 2.0.x)의 Agent 수명주기를
가능한 한 그대로 옮겼다.

충실히 재현한 것:
  - FixedUpdate 0.02s, DecisionPeriod 20, TakeActionsBetweenDecisions = true
  - 관측 120차원(LOCAL 64 + GLOBAL 56), 행동 [9, 9], 보상 함수 전부
  - 승리 조건, 함선 정지, 목격/은닉 판정, 함장 사격, 역할 배정, 평가 모드(규칙봇 고정)
  - ML-Agents 수명주기:
      * episode id(= Python의 agent_id)는 LazyInitialize에서만 발급 → 죽어서 비활성화됐다가
        다시 활성화되면 새 id
      * 비활성화(OnDisable) 시 terminal을 즉시 큐에 넣고, 다음 Academy step에 decision 없이
        terminal만 단독 전송될 수 있음
      * EndEpisode 후 storedActions 초기화([0,0]) → 다음 decision까지 [0,0] 행동이 반복 실행
      * Python env.reset() → Academy.ForcedFullReset: 에이전트 OnEpisodeBegin만 호출,
        GameManager는 리셋 이벤트를 구독하지 않으므로 게임 상태는 그대로

근사한 것 (실제 Unity와 다를 수 있음):
  - NavMesh 대신 직선 이동 (벽/우회 없음, 가속 없음)
  - 방 좌표는 씬 파일에서 추출, 카페 스폰은 NPC 초기 위치 근처로 근사
  - 규칙봇(NPCAIBrain)은 핵심 휴리스틱만 단순화해서 재현
  - Update와 FixedUpdate를 같은 0.02s 틱에서 함께 처리
"""

import math
import os
import random

import numpy as np

DT = 0.02
DECISION_PERIOD = 20

# 씬(SampleScene.unity)에서 추출한 방 좌표 (x, z), roomIndex 순
ROOMS = [
    ("NavigationRoom", (-1.34, 42.12)),
    ("ElectronicRoom", (-68.5, -30.1)),
    ("CommunityRoom", (73.6, -27.5)),
    ("OxygenRoom", (-68.75, -81.6)),
    ("Storage", (1.4, -98.15)),
    ("ProtectorRoom", (74.0, -78.1)),
    ("EngineRoom", (-50.4, -150.06)),
    ("NeclearReactor", (59.6, -151.47)),
]
CAFE = (-10.0, -30.0)
NUM_NPC = 6
MAP_SIZE = 50.0
POS_Y = 1.0

MAX_ROOMS = 8
MAX_OTHER = 6
SHOOT_OFFSET = 3

DEFAULT_PARAMS = {
    "total_distance": 1000.0,
    "ship_speed": 10.0,
    "ship_stop_threshold": 50.0,
    "sabotage_win_threshold": 30.0,
    "sabotage_damage": 10.0,
    "repair_amount": 15.0,
    "sabotage_time": 3.0,
    "repair_time": 5.0,
    "move_speed": 5.0,
    "vision_range": 10.0,
    "max_episode_time": 300.0,
    # 밸런스 옵션 (기본값 = 현재 Unity 동작)
    "room_destroy_loss": 1.0,          # 1: 방 하나 0% → 즉시 사보타주 승리 / 0: 평균 안정도로만 판정
    "alert_threshold": 30.0,           # 위치 알림(전원에게 안정도 공개) 임계값
    "repair_interrupts_sabotage": 0.0, # 1: 부수는 중인 방에 수리를 시작하면 부수기 중단
    "shoot_grace_time": 0.0,           # 게임 시작 후 이 시간(초) 동안 사격 불가
    "share_witness": 0.0,              # 1: 인간이 부수기를 목격하면 함장에게 공유(신고) → 함장 목격 플래그
    "shoot_requires_evidence": 0.0,    # 1: 함장은 목격 플래그(직접 목격/신고)가 있는 대상만 쏠 수 있음
    "captain_bullets": 2.0,
    "shoot_range": -1.0,               # 사격 사거리: -1 = 시야(vision_range), 0 = 무제한(소집 후 처형), >0 = 그 거리
    # 인간팀 보상 셰이핑 (기본 0 = 없음)
    "reward_alert_approach": 0.0,      # 알림 방까지 거리가 1 줄 때마다 보상 (잠재 기반)
    "reward_interrupt": float(os.environ.get("SIM_INTERRUPT_BONUS", "0")),  # 부수기를 끊은 인간팀
    "reward_witness": 0.0,             # 사보타주를 처음 목격한 인간팀 (사보타주당 1회)
    "reward_captain_hit": 0.0,         # 함장이 사보타주를 맞힘
    # 학습 상대 규칙봇: 0 = 없음, 1 = 사보타주 팀을 규칙봇으로, 2 = 인간팀을 규칙봇으로 (평가와 별개)
    "bot_team": 0.0,
    "eval_mode": 0.0,
    "frozen_team": 0.0,
}

# NPCAgent 보상 설정
WIN_REWARD = 5.0
LOSE_REWARD = -5.0
SAB_HIDDEN = 1.0
SAB_WITNESSED = 0.0
REPAIR_REWARD = 0.5
HEALTH_SCALE = 0.01

HUMAN, SABOTEUR, CAPTAIN = 0, 1, 2

# 실험용 변형 (기본값 = 현재 Unity 코드와 동일)
#   SIM_COMMIT_MOVE=1 : 방으로 이동 중에는 새 방 선택을 무시 (도착해야 다음 이동 명령 반영)
COMMIT_MOVE = os.environ.get("SIM_COMMIT_MOVE", "0") == "1"
#   SIM_INTERRUPT_BONUS=x : reward_interrupt 기본값 (이전 실험 E 재현용)


class Room:
    def __init__(self, idx, name, pos):
        self.idx = idx
        self.name = name
        self.pos = np.array(pos, dtype=np.float64)
        self.health = 100.0
        self.alerted = False
        self.used_by = None

    def reset(self):
        self.health = 100.0
        self.alerted = False
        self.used_by = None


class AgentState:
    """ML-Agents Agent 내부 상태 (m_EpisodeId, m_Info.done, m_Reward, storedActions ...)"""

    def __init__(self):
        self.initialized = False
        self.episode_id = -1
        self.done = False
        self.reward = 0.0
        self.stored = [0, 0]
        self.request_decision = False
        self.request_action = False
        self.prev_avg_health = 100.0
        self.prev_alert = None           # (방 번호, 거리) — 알림 방 접근 셰이핑용


class Char:
    def __init__(self, idx):
        self.idx = idx
        self.name = f"NPC {idx + 1}"
        self.pos = np.array(CAFE, dtype=np.float64)
        self.active = True
        self.dead = False
        self.state = "idle"          # idle / moving / interacting
        self.dest = None
        self.room = None             # currentInteractionPoint
        self.interacting = False
        self.timer = 0.0
        self.sabotaging = False
        self.seen = False
        self.use_ml = True
        self.bullets = 2
        self.agent = AgentState()
        # 규칙봇 상태
        self.bot_idle = 0.0
        self.bot_wants_sab = False


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TheThingWorld:
    """Unity 씬 + Academy를 흉내내는 월드. UnityEnvironment(가짜)가 이걸 구동한다."""

    def __init__(self, seed=0, stats_sink=None, game_log=None):
        self.rng = random.Random(seed)
        self.params = dict(DEFAULT_PARAMS)
        self.env_params = {}             # Python에서 받은 environment_parameters
        self.stats_sink = stats_sink     # StatsSideChannel로 보낼 dict
        self.game_log = game_log         # 게임 단위 ground-truth 기록 (리스트)
        self.rooms = [Room(i, n, p) for i, (n, p) in enumerate(ROOMS)]
        self.chars = [Char(i) for i in range(NUM_NPC)]
        self.next_episode_id = 0
        self.academy_step = 0
        self.total_ticks = 0
        self.total_agent_decisions = 0
        self.pending = []                # 이번 exchange로 보낼 (agent_id, obs, reward, done)
        self.game_index = 0
        self._init_game()
        for c in self.chars:
            self._lazy_init(c)

    # ------------------------------------------------------------------
    # 파라미터
    # ------------------------------------------------------------------
    def p(self, k):
        return self.params[k]

    def load_env_params(self):
        for k, v in self.env_params.items():
            if k in self.params:
                self.params[k] = float(v)
        self.eval_mode = self.params["eval_mode"] > 0.5
        self.frozen_team = int(self.params["frozen_team"])
        self.bot_team = int(self.params["bot_team"])

    # ------------------------------------------------------------------
    # 게임 초기화 / 리셋 (GameManager.InitializeGame / ResetGame)
    # ------------------------------------------------------------------
    def _init_game(self):
        self.eval_mode = False
        self.frozen_team = 0
        self.bot_team = 0
        self._reset_state(load_params=False)

    def _reset_state(self, load_params=True):
        if load_params:
            self.load_env_params()
        self.state = "Playing"
        self.game_over = False
        self.distance = 0.0
        self.ship_stopped = False
        self.timer = 0.0
        self.shots = 0
        self.shots_hit = 0
        self.sab_total = 0
        self.sab_hidden = 0
        self.repair_total = 0
        self.sab_interrupted = 0
        self.witness_events = 0
        self.witnessed = {}
        self.alive = list(self.chars)          # allCharacters (사망 시 제거)
        for r in self.rooms:
            r.reset()
        self._update_avg_health()
        for c in self.chars:
            c.active = True
            c.pos = np.array([CAFE[0] + self.rng.uniform(-2, 2), CAFE[1] + self.rng.uniform(-2, 2)])
            c.dead = False
            c.sabotaging = False
            c.interacting = False
            c.state = "idle"
            c.timer = 0.0
            c.room = None
            c.dest = None
            c.bullets = int(self.p("captain_bullets"))
            c.bot_idle = 0.0
            c.bot_wants_sab = False
        self.suspicion = {}
        self._exec_timer = 0.0
        # 역할 배정 (RoleManager.AssignRoles)
        order = list(self.chars)
        self.rng.shuffle(order)
        self.role = {}
        for c in order[:2]:
            self.role[c.idx] = SABOTEUR
        rest = order[2:]
        cap = rest.pop(self.rng.randrange(len(rest)))
        self.role[cap.idx] = CAPTAIN
        for c in rest:
            self.role[c.idx] = HUMAN
        self.saboteurs = [c for c in self.chars if self.role[c.idx] == SABOTEUR]
        self.humans = [c for c in self.chars if self.role[c.idx] == HUMAN]
        self.captain = cap
        # ApplyControlModes
        for c in self.chars:
            rule = False
            is_sab = self.role[c.idx] == SABOTEUR
            if self.eval_mode and self.frozen_team != 0:
                # 1=사보타주 봇, 2=인간팀 봇, 3=전원 봇(시뮬레이터 분석 전용)
                rule = True if self.frozen_team == 3 else (is_sab if self.frozen_team == 1 else (not is_sab))
            elif self.bot_team in (1, 2):
                # 학습 커리큘럼: 한 팀을 규칙봇 상대로
                rule = is_sab if self.bot_team == 1 else (not is_sab)
            c.use_ml = not rule
        self.game_index += 1
        self.game_start_tick = self.total_ticks
        self.game_eval_mode = self.eval_mode
        self.game_frozen = self.frozen_team
        self.game_bot_team = self.bot_team if not self.eval_mode else 0

    # ------------------------------------------------------------------
    # ML-Agents Agent 수명주기
    # ------------------------------------------------------------------
    def _lazy_init(self, c):
        a = c.agent
        if a.initialized:
            return
        a.initialized = True
        a.episode_id = self.next_episode_id
        self.next_episode_id += 1
        a.done = False
        a.stored = [0, 0]
        if self.total_ticks != 0:
            self._on_episode_begin(c)

    def _on_episode_begin(self, c):
        c.agent.prev_avg_health = self.avg_health
        c.agent.prev_alert = None

    def _notify_done(self, c, disabled):
        a = c.agent
        if a.done:
            return
        a.done = True
        self.pending.append((a.episode_id, self._collect_obs(c), a.reward, True))
        a.reward = 0.0
        a.request_action = False
        a.request_decision = False
        a.stored = [0, 0]

    def _end_episode(self, c):
        self._notify_done(c, disabled=False)
        c.agent.stored = [0, 0]           # _AgentReset → ResetData
        self._on_episode_begin(c)

    def _set_active(self, c, active):
        if c.active == active:
            return
        c.active = active
        if not active:
            self._notify_done(c, disabled=True)   # OnDisable
            c.agent.initialized = False
        else:
            self._lazy_init(c)                    # OnEnable

    def force_reset(self):
        """Python env.reset() → Academy.ForcedFullReset (게임은 리셋되지 않음)"""
        self.academy_step = 0
        for c in self.chars:
            if c.active and c.agent.initialized:
                c.agent.stored = [0, 0]
                self._on_episode_begin(c)

    # ------------------------------------------------------------------
    # Academy step 루프
    # ------------------------------------------------------------------
    def advance(self, actions_by_id):
        """
        Python이 넘긴 행동을 적용하고, 다음 exchange가 필요해질 때까지 시뮬레이션.
        반환: [(agent_id, obs, reward, done), ...]
        """
        for c in self.chars:
            if c.active and c.agent.episode_id in actions_by_id:
                c.agent.stored = list(actions_by_id[c.agent.episode_id])
        while True:
            # AgentAct
            for c in self.chars:
                if c.active and c.agent.initialized and c.agent.request_action:
                    c.agent.request_action = False
                    self._on_action_received(c, c.agent.stored)
            self.academy_step += 1
            # Update (게임 로직)
            self._update()
            self.total_ticks += 1
            # AgentPreStep (DecisionRequester)
            for c in self.chars:
                if c.active and c.agent.initialized:
                    if self.academy_step % DECISION_PERIOD == 0:
                        c.agent.request_decision = True
                    c.agent.request_action = True
            # AgentSendState
            for c in self.chars:
                a = c.agent
                if c.active and a.initialized and a.request_decision:
                    a.done = False
                    self.pending.append((a.episode_id, self._collect_obs(c), a.reward, False))
                    a.reward = 0.0
                    a.request_decision = False
            # DecideAction → 큐가 비어있지 않으면 Python과 통신
            if self.pending:
                out = self.pending
                self.pending = []
                self.total_agent_decisions += sum(1 for o in out if not o[3])
                return out

    def first_exchange(self):
        """초기 reset 직후: step 0에서 모든 에이전트가 decision 요청"""
        for c in self.chars:
            if c.active and c.agent.initialized:
                c.agent.request_decision = True
                c.agent.request_action = True
        for c in self.chars:
            a = c.agent
            if c.active and a.initialized and a.request_decision:
                a.done = False
                self.pending.append((a.episode_id, self._collect_obs(c), a.reward, False))
                a.reward = 0.0
                a.request_decision = False
        out = self.pending
        self.pending = []
        self.total_agent_decisions += sum(1 for o in out if not o[3])
        return out

    # ------------------------------------------------------------------
    # 관측 (NPCAgent.CollectObservations)
    # ------------------------------------------------------------------
    def _other_slots(self, c):
        return [o for o in self.chars if o is not c]

    def alive_human_ratio(self):
        tot = alive = 0
        for o in self.chars:
            if self.role[o.idx] != SABOTEUR:
                tot += 1
                if o.active:
                    alive += 1
        return alive / tot if tot else 0.0

    def _collect_obs(self, c):
        vis = self.p("vision_range")
        is_sab = 1.0 if self.role[c.idx] == SABOTEUR else 0.0
        is_cap = 1.0 if self.role[c.idx] == CAPTAIN else 0.0
        pos3 = [c.pos[0] / MAP_SIZE, POS_Y / MAP_SIZE, c.pos[1] / MAP_SIZE]
        game = [self.avg_health / 100.0, self.distance / self.p("total_distance"), self.alive_human_ratio()]
        local = [is_sab, is_cap] + pos3 + game
        glob = [is_sab, is_cap] + pos3 + game
        for r in self.rooms:
            d = dist(c.pos, r.pos)
            local.append(d / MAP_SIZE)
            if d <= vis or r.health <= self.p("alert_threshold"):
                local += [r.health / 100.0, 1.0 if r.used_by is not None else 0.0, 1.0]
            else:
                local += [0.0, 0.0, 0.0]
            glob += [d / MAP_SIZE, r.health / 100.0, 1.0 if r.used_by is not None else 0.0]
        seen = self.witnessed.get(c.idx, ())
        for o in self._other_slots(c):
            alive = o.active
            visible = alive and dist(c.pos, o.pos) <= vis
            local += [o.pos[0] / MAP_SIZE if visible else 0.0,
                      o.pos[1] / MAP_SIZE if visible else 0.0,
                      1.0 if visible else 0.0,
                      1.0 if (alive and o.idx in seen) else 0.0]
            glob += [o.pos[0] / MAP_SIZE if alive else 0.0,
                     o.pos[1] / MAP_SIZE if alive else 0.0,
                     1.0 if alive else 0.0,
                     1.0 if (alive and self.role[o.idx] == SABOTEUR) else 0.0]
        pad = MAX_OTHER - len(self._other_slots(c))   # 빈 슬롯은 0 (C#: GetOtherSlot == null)
        local += [0.0] * (4 * pad)
        glob += [0.0] * (4 * pad)
        obs = np.array(local + glob, dtype=np.float32)
        assert obs.shape == (120,), obs.shape
        return obs

    # ------------------------------------------------------------------
    # 행동 (NPCAgent.OnActionReceived)
    # ------------------------------------------------------------------
    def _on_action_received(self, c, act):
        if self.game_over or not c.use_ml:
            return
        room_choice, inter = int(act[0]), int(act[1])
        role = self.role[c.idx]
        if inter >= SHOOT_OFFSET and role == CAPTAIN:
            self._try_shoot(c, inter - SHOOT_OFFSET)
            if self.game_over or not c.active:
                return
        if not c.interacting and room_choice < MAX_ROOMS and not (COMMIT_MOVE and c.state == "moving"):
            self._move_to_room(c, self.rooms[room_choice])
        if inter in (1, 2) and not c.interacting and self._near(c):
            if inter == 1:
                if role == SABOTEUR:
                    self._try_start(c, True)
            else:
                self._try_start(c, False)
        # CalculateStepReward
        a = c.agent
        delta = self.avg_health - a.prev_avg_health
        a.reward += (-delta if role == SABOTEUR else delta) * HEALTH_SCALE
        a.prev_avg_health = self.avg_health
        # 알림 방 접근 셰이핑 (인간팀): 가장 가까운 알림 방까지 거리가 줄어든 만큼 보상
        k = self.p("reward_alert_approach")
        if k and role != SABOTEUR:
            alerted = [r for r in self.rooms if r.alerted]
            if alerted:
                r = min(alerted, key=lambda r: dist(c.pos, r.pos))
                d = dist(c.pos, r.pos)
                if a.prev_alert is not None and a.prev_alert[0] == r.idx:
                    a.reward += k * (a.prev_alert[1] - d)
                a.prev_alert = (r.idx, d)
            else:
                a.prev_alert = None

    def _try_shoot(self, c, slot):
        if c.bullets <= 0 or self.timer < self.p("shoot_grace_time"):
            return
        others = self._other_slots(c)
        if slot >= len(others):
            return
        t = others[slot]
        if not t.active:
            return
        if not self._can_shoot_target(c, t):
            return
        self._execute(c, t)

    def _can_shoot_target(self, c, t):
        """사격 규칙: 근거(목격 플래그) + 사거리"""
        if self.p("shoot_requires_evidence") > 0.5 and t.idx not in self.witnessed.get(c.idx, ()):
            return False
        rng = self.p("shoot_range")
        if rng < 0:
            rng = self.p("vision_range")
        return rng == 0 or dist(c.pos, t.pos) <= rng

    def _execute(self, shooter, t):
        """CaptainGun.TryExecuteTarget"""
        shooter.bullets -= 1
        was_sab = self.role[t.idx] == SABOTEUR
        self.shots += 1
        if was_sab:
            self.shots_hit += 1
            if shooter.use_ml:
                shooter.agent.reward += self.p("reward_captain_hit")
        self._die(t)
        hc, sc = self.alive_human_count(), len(self.saboteurs)
        if was_sab:
            if sc == 0:
                self.human_win("모든 사보타주 제거")
        else:
            if hc <= sc:
                self.sabotage_win("인간과 동수")

    def _die(self, t):
        t.dead = True
        if t.interacting and t.room is not None and t.room.used_by is t:
            t.room.used_by = None
        t.agent.reward += -1.0                  # NPCAgent.OnDeath
        if t is not self.captain:               # RemoveCharacter
            if t in self.alive:
                self.alive.remove(t)
            if t in self.saboteurs:
                self.saboteurs.remove(t)
            if t in self.humans:
                self.humans.remove(t)
        self._set_active(t, False)

    def alive_human_count(self):
        return len(self.humans) + (1 if self.captain is not None else 0)

    # ------------------------------------------------------------------
    # NPCController
    # ------------------------------------------------------------------
    def _move_to_room(self, c, room):
        if c.interacting:
            self._cancel(c)
        c.room = room
        c.dest = room.pos.copy()
        c.state = "moving"

    def _cancel(self, c):
        if not c.interacting:
            return
        if c.room is not None and c.room.used_by is c:
            c.room.used_by = None
        c.interacting = False
        c.timer = 0.0
        c.sabotaging = False
        c.room = None
        c.state = "idle"

    def _near(self, c):
        return c.room is not None and dist(c.pos, c.room.pos) <= 2.0

    def _try_start(self, c, sabotage):
        if c.room is None or not self._near(c):
            return False
        if c.room.used_by is not None:
            u = c.room.used_by
            # 인간팀(선원/함장)만 끊을 수 있음. 사보타주가 "수리"로 동료의 부수기를 끊는 버그 수정
            if (not sabotage and u is not c and u.sabotaging
                    and self.role[c.idx] != SABOTEUR
                    and self.p("repair_interrupts_sabotage") > 0.5):
                self._cancel(u)                  # 수리 시작이 부수기를 끊는다
                self.sab_interrupted += 1
                if c.use_ml:
                    c.agent.reward += self.p("reward_interrupt")
            else:
                return False
        c.room.used_by = c
        c.sabotaging = sabotage
        c.interacting = True
        c.timer = 0.0
        c.state = "interacting"
        c.seen = self._witnesses(c, record=True) > 0 if sabotage else False
        return True

    def _witnesses(self, sab, record):
        vis = self.p("vision_range")
        n = 0
        for o in self.alive:
            if o is sab or not o.active or self.role[o.idx] == SABOTEUR:
                continue
            if dist(sab.pos, o.pos) <= vis:
                n += 1
                if record:
                    if sab.idx not in self.witnessed.get(o.idx, ()):
                        self.witness_events += 1
                        if o.use_ml:
                            o.agent.reward += self.p("reward_witness")
                    self.witnessed.setdefault(o.idx, set()).add(sab.idx)
                    cap = self.captain
                    if (self.p("share_witness") > 0.5 and cap is not None and cap.active
                            and cap is not o):
                        self.witnessed.setdefault(cap.idx, set()).add(sab.idx)
        return n

    def _complete(self, c):
        room = c.room
        before = room.health
        if c.sabotaging:
            room.health = max(0.0, room.health - self.p("sabotage_damage"))
            if room.health <= self.p("alert_threshold") and not room.alerted:
                room.alerted = True
                self._on_alert(room)
            self._on_sabotage_detected(c, room)
        else:
            room.health = min(100.0, room.health + self.p("repair_amount"))
            if room.health > self.p("alert_threshold") and room.alerted:
                room.alerted = False
        change = room.health - before
        self._update_avg_health()
        room.used_by = None
        role = self.role[c.idx]
        if c.sabotaging:
            hidden = not c.seen
            self.sab_total += 1
            if hidden:
                self.sab_hidden += 1
            if role == SABOTEUR and c.use_ml:
                c.agent.reward += SAB_HIDDEN if hidden else SAB_WITNESSED
        else:
            if change > 0:
                self.repair_total += 1
                if role != SABOTEUR and c.use_ml:
                    c.agent.reward += REPAIR_REWARD * min(1.0, change / self.p("repair_amount"))
        c.interacting = False
        c.timer = 0.0
        c.sabotaging = False
        c.room = None
        c.state = "idle"
        c.bot_idle = self.rng.uniform(1.0, 3.0)

    def _update_avg_health(self):
        self.avg_health = sum(r.health for r in self.rooms) / len(self.rooms)
        thr = self.p("ship_stop_threshold")
        if self.avg_health <= thr and not self.ship_stopped:
            self.ship_stopped = True
        elif self.avg_health > thr and self.ship_stopped:
            self.ship_stopped = False

    # ------------------------------------------------------------------
    # Update (프레임 로직)
    # ------------------------------------------------------------------
    def _update(self):
        if self.state != "Playing":
            return
        speed = self.p("move_speed")
        for c in self.chars:
            if not c.active or c.dead:
                continue
            if not c.use_ml:
                self._bot_think(c)
            if c.state == "moving" and c.dest is not None:
                d = c.dest - c.pos
                remain = math.hypot(d[0], d[1])
                step = speed * DT
                if remain <= step:
                    c.pos = c.dest.copy()
                    remain = 0.0
                else:
                    c.pos = c.pos + d / remain * step
                    remain -= step
                if remain <= 0.5:
                    if c.use_ml:
                        c.state = "idle"
                    elif c.room is not None:
                        if not self._try_start(c, self.role[c.idx] == SABOTEUR and c.bot_wants_sab):
                            c.state = "idle"
                            c.room = None
                    else:
                        c.state = "idle"
            elif c.state == "interacting":
                c.timer += DT
                if c.sabotaging and self._witnesses(c, record=True) > 0:
                    c.seen = True
                need = self.p("sabotage_time") if c.sabotaging else self.p("repair_time")
                if c.timer >= need:
                    self._complete(c)
                    if self.game_over:
                        return
        if self.captain is not None and self.captain.active and not self.captain.use_ml:
            self._captain_bot(self.captain)
            if self.game_over:
                return
        # GameManager.Update
        self.distance += (0.0 if self.ship_stopped else self.p("ship_speed")) * DT
        self.timer += DT
        if self.timer >= self.p("max_episode_time"):
            self.sabotage_win("에피소드 시간 초과")
            return
        # SystemHealth.CheckWinConditions
        if self.p("room_destroy_loss") > 0.5:
            for r in self.rooms:
                if r.health <= 0.0:
                    self.sabotage_win(f"{r.name} 완전 파괴")
                    return
        if self.avg_health <= self.p("sabotage_win_threshold"):
            self.sabotage_win("평균 안정도 임계 이하")
            return
        if self.distance / self.p("total_distance") >= 1.0:
            self.human_win("목적지 도착")
            return
        hc, sc = self.alive_human_count(), len(self.saboteurs)
        if hc <= sc and sc > 0:
            self.sabotage_win("인간과 동수")
            return
        if sc == 0:
            self.human_win("모든 사보타주 제거")

    # ------------------------------------------------------------------
    # 게임 종료 (GameManager.NotifyGameEnd)
    # ------------------------------------------------------------------
    def human_win(self, reason):
        if self.game_over:
            return
        self.game_over = True
        self._notify_game_end(True, reason)

    def sabotage_win(self, reason):
        if self.game_over:
            return
        self.game_over = True
        self._notify_game_end(False, reason)

    def _stat(self, key, val):
        if self.stats_sink is not None:
            self.stats_sink.setdefault(key, []).append((float(val), 0))

    def _notify_game_end(self, human_win, reason):
        if self.shots > 0:
            self._stat("behavior/captain_shot_accuracy", self.shots_hit / self.shots)
        if self.sab_total > 0:
            self._stat("behavior/sabotage_hidden_rate", self.sab_hidden / self.sab_total)
        self._stat("behavior/sabotage_count", self.sab_total)
        self._stat("behavior/repair_count", self.repair_total)
        self._stat("outcome/human_win", 1.0 if human_win else 0.0)
        self._stat("train/bot_team", float(self.game_bot_team))
        if self.eval_mode and self.frozen_team == 2:
            self._stat("eval/saboteur_vs_bot_win", 0.0 if human_win else 1.0)
        elif self.eval_mode and self.frozen_team == 1:
            self._stat("eval/human_vs_bot_win", 1.0 if human_win else 0.0)

        if self.game_log is not None:
            self.game_log.append({
                "game": self.game_index,
                "agent_decisions": self.total_agent_decisions,
                "human_win": int(human_win),
                "reason": reason,
                "duration": round(self.timer, 2),
                "eval_mode": int(self.game_eval_mode),
                "frozen_team": self.game_frozen,
                "bot_team": self.game_bot_team,
                "sab_total": self.sab_total,
                "sab_hidden": self.sab_hidden,
                "repairs": self.repair_total,
                "sab_interrupted": self.sab_interrupted,
                "witness_events": self.witness_events,
                "captain_flags": len(self.witnessed.get(self.captain.idx, ())) if self.captain else 0,
                "shots": self.shots,
                "shots_hit": self.shots_hit,
                "avg_health": round(self.avg_health, 1),
                "progress": round(self.distance / self.p("total_distance"), 3),
            })

        dead = [c for c in self.chars if not c.active]
        for c in dead:
            self._set_active(c, True)            # 1. 죽은 에이전트 재활성화
        for c in self.chars:                     # 2. 보상
            role = self.role[c.idx]
            if role == SABOTEUR:
                c.agent.reward += LOSE_REWARD if human_win else WIN_REWARD
            else:
                c.agent.reward += WIN_REWARD if human_win else LOSE_REWARD
        self._reset_state()                      # 3. ResetGame (역할 재배정 포함)
        for c in self.chars:                     # 4. EndEpisode
            self._end_episode(c)

    # ------------------------------------------------------------------
    # 규칙봇 (NPCAIBrain 단순화)
    # ------------------------------------------------------------------
    def _visible_count_near(self, pos, me):
        vis = self.p("vision_range")
        return sum(1 for o in self.alive if o is not me and o.active
                   and dist(o.pos, pos) <= vis)

    def _bot_think(self, c):
        if c.interacting or c.state == "moving":
            return
        c.bot_idle -= DT
        if c.bot_idle > 0:
            return
        role = self.role[c.idx]
        if role == SABOTEUR:
            # (손상도+0.1)^2 × 1/거리 × 안전도 를 softmax로 선택
            scores = []
            for r in self.rooms:
                if r.used_by is not None:
                    scores.append(-1e9)
                    continue
                dmg = (100.0 - r.health) / 100.0
                dd = max(dist(c.pos, r.pos), 1.0)
                safety = 1.0 / (1.0 + self._visible_count_near(r.pos, c)) ** 2
                scores.append(math.log((dmg + 0.1) ** 2 / dd * safety * 1000.0 + 1e-9))
            m = max(scores)
            w = [math.exp((s - m) / 0.5) for s in scores]
            r = self.rng.choices(self.rooms, weights=w)[0]
            c.bot_wants_sab = True
        else:
            # 알람 방 우선, 없으면 다른 캐릭터와 먼 방으로 분산
            can_cut = self.p("repair_interrupts_sabotage") > 0.5
            alerted = [r for r in self.rooms if r.alerted and (r.used_by is None or can_cut)]
            if alerted:
                r = min(alerted, key=lambda r: dist(c.pos, r.pos))
            else:
                cand = [r for r in self.rooms if r.used_by is None]
                def disp(r):
                    others = [o for o in self.alive if o is not c and o.active]
                    if not others:
                        return 0.0
                    return min(dist(o.pos, r.pos) for o in others)
                w = [disp(r) + 1.0 for r in cand]
                r = self.rng.choices(cand, weights=w)[0]
            c.bot_wants_sab = False
        c.room = r
        c.dest = r.pos.copy()
        c.state = "moving"

    def _on_alert(self, room):
        cap = self.captain
        if cap is None or cap.use_ml:
            return
        for o in self.alive:
            if o is cap or not o.active:
                continue
            if dist(o.pos, room.pos) <= 15.0:
                self.suspicion[o.idx] = self.suspicion.get(o.idx, 0.0) + 30.0

    def _on_sabotage_detected(self, sab, room):
        cap = self.captain
        if cap is None or cap.use_ml or sab is cap:
            return
        vis = self.p("vision_range")
        if dist(cap.pos, sab.pos) <= vis or (self.p("share_witness") > 0.5 and sab.seen):
            self.suspicion[sab.idx] = self.suspicion.get(sab.idx, 0.0) + 50.0
        else:
            for o in self.alive:
                if o is cap or not o.active:
                    continue
                if (dist(o.pos, room.pos) <= 10.0
                        and dist(cap.pos, o.pos) <= vis):
                    self.suspicion[o.idx] = self.suspicion.get(o.idx, 0.0) + 20.0

    def _captain_bot(self, cap):
        if cap.bullets <= 0 or self.timer < self.p("shoot_grace_time"):
            return
        if self.p("shoot_requires_evidence") > 0.5:
            # 근거 규칙: 목격(신고 포함)한 대상 중 사거리 안에 있는 대상만
            cands = [(1.0, i) for i in self.witnessed.get(cap.idx, ())
                     if self.chars[i].active and self._can_shoot_target(cap, self.chars[i])]
        else:
            cands = [(v, i) for i, v in self.suspicion.items()
                     if v >= 100.0 and self.chars[i].active]
        if not cands:
            self._exec_timer = 0.0
            return
        self._exec_timer += DT
        if self._exec_timer >= 3.0:
            _, i = max(cands)
            self.suspicion.pop(i, None)
            self._exec_timer = 0.0
            self._execute(cap, self.chars[i])
