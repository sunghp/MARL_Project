# 최종 Unity 적용 안내 (사보타주 vs 규칙봇 인간팀)

**현재 방향: 인간팀(선원·함장)은 강화 규칙봇으로 고정하고, 사보타주 2명만 학습한다.** (`sim/results/saboteur_vs_bot/REPORT.md`)

- 가상 환경 결과: 학습 사보타주(`models/saboteur_vs_bot_8M.pt`)는 규칙봇 인간팀을 41% 이긴다.
  - 스크립트 전략(방 붙기 17%, 회피 16%)보다 훨씬 높다.
  - 사람이 짠 규칙봇 사보타주(42%, 인간팀 위치를 모두 아는 특권 있음)와는 동률이다.
- 이 설정은 아래 C# 스크립트, 씬, yaml에 모두 반영돼 있다.
- 이전 방향(두 팀 공동 학습, 밸런스 N)은 `config/TheThing_selfplay_v4.yaml`에 남아 있다. `sim/results/balance_v4_4m`이 그 결과다.

## 스크립트 (`Assets/Scripts/`)

| 파일 | 역할 / 이번 작업에서 바뀐 것 |
|---|---|
| `NPCAgent.cs` | ML 에이전트. 관측 120차원(Actor 64 + Critic 56), 행동 [9, 9](방 선택 / 없음·부수기·수리·슬롯 사격 6개). 고정 슬롯 관측, 목격 플래그, 이동 확정, 근거 기반 사격, 수리 보상(실제 회복량 비례), 셰이핑 훅 |
| `GameManager.cs` | 게임 진행·리셋·승패. 모든 밸런스 파라미터와 env param 연동, 승패/평가 통계(`outcome/human_win`, `eval/*`, `train/bot_team`), 목격 기록·공유, `CanShootTarget`, 분산 스폰, 함선 속도 ∝ 안정도, 학습 중 Player 제외. **강화 봇: `botLevel` 등 파라미터, 인간팀 공유 정보(`IsKnownDamaged`)** |
| `NPCController.cs` | 이동·상호작용. 자동 수리(`TryAutoRepair`, 강화 봇도 사용), `IsMovingToRoom`, 리셋 시 이동 속도 재적용, 실제 회복량 기반 수리 집계. **강화 봇: 이동 중 재지정, 도착 규칙, 함장 추적(`ChaseTarget`)** |
| `InteractionPoint.cs` | 방. 알림 임계값을 GameManager 값으로, 인간팀이 수리를 시작하면 부수기 끊기, 실제 안정도 변화량 반환 |
| `SystemHealth.cs` | 승리 조건. `roomDestroyLoss` 옵션 |
| `CaptainGun.cs` | 처형. Player 대상 처리, 함장 명중 훅, 시작 직후 사격 유예 |
| `Npcaibrain.cs` | 규칙봇. 사격 유예, 신고(목격 공유) 반영, 근거 규칙. **강화 봇(`botLevel 1`): 인간팀 손상 대응·순찰·재지정(`DecideAsCrewV1`, `ShouldRetargetV1`), 함장 추적(`UpdateChase`), 회피형 사보타주(`DecideAsSaboteurV1`)** |
| `RoleManager.cs`, `PlayerController.cs`, `Spectate.cs` | 변경 없음 |

## 최종 값

`config/TheThing.yaml`의 `environment_parameters`가 학습 시 Unity로 주입된다. GameManager 코드 기본값과 씬 값도 같게 맞춰 두었다.

| 항목 | 값 |
|---|---|
| 학습 상대 `bot_team` | **2** (인간팀은 항상 규칙봇, 사보타주만 학습) |
| 규칙봇 수준 `bot_level` | **1** (강화 봇) |
| 강화 봇 변형 `bot_perception` / `bot_reaction` / `bot_chase` | 1 / 0초 / 켬 |
| 알림 임계값 `alert_threshold` | **60** |
| 수리로 부수기 끊기 `repair_interrupts_sabotage` | 켬 (인간팀만) |
| 시작 후 사격 유예 `shoot_grace_time` | 10초 |
| 목격 공유 `share_witness` | 켬 |
| 시야 `vision_range` | **10** |
| 근거 사격 `shoot_requires_evidence` / 사거리 `shoot_range` / 총알 | 켬 / **15** / 2 |
| 이동 확정 `commit_move` | 켬 |
| 자동 수리 `auto_repair` | 켬 |
| 분산 스폰 `spawn_spread` | 켬 |
| 함선 속도 ∝ 평균 안정도 `ship_speed_by_health` | 켬 |
| 부수기 피해 `sabotage_damage` | **8** |
| 방 하나 0% → 즉시 패배 `room_destroy_loss` | 켬 |
| 보상 셰이핑 `reward_*` | 0 (끔) |
| 트레이너: 사보타주 엔트로피 계수 | **0.003** (`--entropy-role saboteur=0.003`, 기본 0.01로는 4M에서 6~12%) |
| 사보타주 관측 v2 `sab_obs` / 목격 페널티 `reward_sab_witnessed` | 0 / 0 (실험했지만 이득 없음, `sim/results/saboteur_improve`. 최종 모델은 v1 관측으로 학습됨) |

## Unity 설정 체크리스트

이 저장소의 프리팹과 씬은 이미 맞춰져 있다. 다른 프로젝트로 스크립트만 옮길 때 확인할 것:

- NPC `Behavior Parameters`: Vector Observation **120**, Discrete Branches **[9, 9]**, Behavior Name `TheThing`
- NPC `Decision Requester`: Decision Period **20**, Take Actions Between Decisions **켬**
- NPC 6명(Tag `NPC`), NavMeshSurface, 방 8개에 `InteractionPoint`(roomIndex 0~7), GameManager의 `cafeSpawnPoint`
- 학습 중(Python 연결 시) Player는 자동으로 제외된다(`includePlayerInTraining`)

## 그냥 Play를 누르면? → 학습된 정책이 아니다

- 트레이너(Python)가 연결되지 않고 NPC의 `Behavior Parameters`에 ONNX 모델도 없으면, ML-Agents는 `NPCAgent.Heuristic()`으로 대신 움직인다.
  - 기본 행동은 "대기 + 없음"이라 **NPC들이 거의 움직이지 않는다.**
  - 키보드 1~8(방 선택), Q(부수기), E(수리), F(사격)를 누르면 **NPC 전원이 같은 행동**을 한다.
- 이 경우 Player도 게임에 포함된다(학습 중에만 자동 제외).
- GameManager Inspector의 **Bot Team을 2**로 두면 인간팀은 규칙봇이 움직인다. 사보타주만 Heuristic이라 멈춰 있다. 규칙봇 동작을 눈으로 확인할 때 쓴다.
- 이 프로젝트의 트레이너는 PyTorch(`.pt`)로 저장한다. Unity에 바로 넣을 ONNX 모델은 만들지 않는다.

### 학습된 정책을 Unity에서 보는 방법: `spectate.py`

Python이 정책을 계산하고 Unity는 화면만 보여 준다. ONNX 변환이 필요 없다.

```bash
# 에디터로 보기: 아래를 실행한 뒤 Unity 에디터에서 Play
python spectate.py --checkpoint models/saboteur_vs_bot_8M.pt --editor

# 빌드로 보기
python spectate.py --checkpoint <체크포인트.pt> --env-path <빌드 경로>
```

- `models/saboteur_vs_bot_8M.pt`: **가상 환경에서** 규칙봇 인간팀을 상대로 7.9M 스텝 학습한 사보타주 모델이다.
  - yaml의 `bot_team: 2` 때문에 관전할 때도 인간팀은 규칙봇이 움직이고, 사보타주만 이 모델을 쓴다.
  - 관측·행동 규격이 Unity와 같아서 그대로 붙는다. 하지만 Unity에서 학습한 것은 아니다.
- 이전 공동 학습 모델 `models/sim_balance_v4_4M_seed2.pt`는 `--balance-yaml`이 없는 spectate에서는 새 yaml(인간팀 봇)로 돈다. 공동 학습 그대로 보려면 yaml의 `bot_team`을 0으로 바꾼다.
  - NavMesh 경로와 규칙 차이 때문에 시뮬레이터만큼 잘 움직인다는 보장은 없다.
  - 대신 "시뮬레이터 결과가 Unity로 옮겨지는가"를 학습 없이 바로 확인하는 첫 테스트로 쓸 수 있다.
- 행동은 기본적으로 학습 때와 같은 확률적 샘플링이다. `--deterministic`을 주면 argmax를 쓴다. 인간·함장 정책은 거의 균등분포라 argmax는 학습 때와 다르게 보일 수 있다.
- 제대로 된 결과물은 아래 "학습"으로 Unity 빌드에서 직접 학습한 체크포인트를 같은 방법으로 관전하는 것이다.

## 학습 (단계별 안내: `TRAINING_GUIDE.md`)

```bash
pip install torch numpy pyyaml tensorboard mlagents-envs
python mappo_trainer.py --env-path <빌드 경로> --entropy-role saboteur=0.003
```

- `scripts/train.sh`(mlagents-learn / POCA)는 이 구조와 맞지 않으니 쓰지 않는다.
- 사보타주 엔트로피 계수는 0.003으로 낮춘다. 기본 0.01은 공동 학습용 값이다(`sim/results/interrupt_vs_entropy`).
- 학습 곡선은 시드에 따라 크게 다르다(같은 설정 41% / 16%). 가능하면 시드 2개 이상을 돌린다.

## 검증 상태와 한계

- **컴파일**: Unity API 스텁으로 C# 9 컴파일을 통과했다(`tools/csharp_check`). Unity 에디터에서는 아직 열어 보지 않았다.
- **확인 필요**
  - 분산 스폰 위치가 실제 맵에서 걸을 수 있는 곳인지 (`NavMesh.SamplePosition` 보정)
  - 자동 수리, 이동 확정, 근거 사격이 의도대로 동작하는지 (한 판 관전으로 확인)
  - 짧은 학습(약 20만 스텝)에서 에피소드 수 = 게임 수인지, 게임당 부수기·수리가 시뮬레이터와 비슷한지
- **강화 봇 확인**: C# 이식은 시뮬레이터(`sim/thething_sim.py`의 bot_level 1)와 같은 규칙으로 썼다. Unity에서 규칙봇끼리 몇 판 돌려 아래를 확인한다.
  - 사보타주 승률 약 40%, 판당 부수기 약 24회, 처형된 사보타주 약 1.3명
  - 손상 방으로 인간팀이 몰려가는지, 함장이 목격된 사보타주를 쫓는지
- **결과의 한계**: 시뮬레이터는 NavMesh 대신 직선으로 이동한다. 규칙봇 사보타주는 인간팀 위치를 모두 알지만, 학습 사보타주는 자기 시야만 본다.
