# 최종 Unity 적용 안내 (가상 환경에서 가장 좋았던 설정)

가상 환경 실험 중 가장 좋았던 결과는 **새 밸런스 N**이다(`sim/results/balance_v4_long`, `balance_v4_4m`).

- 셰이핑·커리큘럼 없이 4M 스텝, 시드 4개에서 두 팀이 붕괴 없이 함께 학습했다.
- 학습 정책끼리 인간 승률 60~80%, 사보타주는 규칙봇 인간팀 상대 0% → 28%, 함장 근거 사격 명중 100%.

이 설정은 아래 C# 스크립트, 씬, 프리팹, yaml에 모두 반영돼 있다.

## 스크립트 (`Assets/Scripts/`)

| 파일 | 역할 / 이번 작업에서 바뀐 것 |
|---|---|
| `NPCAgent.cs` | ML 에이전트. 관측 120차원(Actor 64 + Critic 56), 행동 [9, 9](방 선택 / 없음·부수기·수리·슬롯 사격 6개). 고정 슬롯 관측, 목격 플래그, 이동 확정, 근거 기반 사격, 수리 보상(실제 회복량 비례), 셰이핑 훅 |
| `GameManager.cs` | 게임 진행·리셋·승패. 모든 밸런스 파라미터와 env param 연동, 승패/평가 통계(`outcome/human_win`, `eval/*`, `train/bot_team`), 목격 기록·공유, `CanShootTarget`, 분산 스폰, 함선 속도 ∝ 안정도, 학습 중 Player 제외 |
| `NPCController.cs` | 이동·상호작용. 자동 수리(`TryAutoRepair`), `IsMovingToRoom`, 리셋 시 이동 속도 재적용, 실제 회복량 기반 수리 집계 |
| `InteractionPoint.cs` | 방. 알림 임계값을 GameManager 값으로, 인간팀이 수리를 시작하면 부수기 끊기, 실제 안정도 변화량 반환 |
| `SystemHealth.cs` | 승리 조건. `roomDestroyLoss` 옵션 |
| `CaptainGun.cs` | 처형. Player 대상 처리, 함장 명중 훅, 시작 직후 사격 유예 |
| `Npcaibrain.cs` | 규칙봇(평가용). 사격 유예, 신고(목격 공유) 반영, 근거 규칙 |
| `RoleManager.cs`, `PlayerController.cs`, `Spectate.cs` | 변경 없음 |

## 최종 값

`config/TheThing.yaml`의 `environment_parameters`가 학습 시 Unity로 주입된다. GameManager 코드 기본값과 씬 값도 같게 맞춰 두었다.

| 항목 | 값 |
|---|---|
| 알림 임계값 `alert_threshold` | 70 |
| 수리로 부수기 끊기 `repair_interrupts_sabotage` | 켬 (인간팀만) |
| 시작 후 사격 유예 `shoot_grace_time` | 10초 |
| 목격 공유 `share_witness` | 켬 |
| 시야 `vision_range` | 12 |
| 근거 사격 `shoot_requires_evidence` / 사거리 `shoot_range` / 총알 | 켬 / 25 / 2 |
| 이동 확정 `commit_move` | 켬 |
| 자동 수리 `auto_repair` | 켬 |
| 분산 스폰 `spawn_spread` | 켬 |
| 함선 속도 ∝ 평균 안정도 `ship_speed_by_health` | 켬 |
| 부수기 피해 `sabotage_damage` | 7 |
| 방 하나 0% → 즉시 패배 `room_destroy_loss` | 켬 |
| 보상 셰이핑 `reward_*` | 0 (끔) |
| 커리큘럼 `bot_team` | 0 (끔) |

## Unity 설정 체크리스트

이 저장소의 프리팹과 씬은 이미 맞춰져 있다. 다른 프로젝트로 스크립트만 옮길 때 확인할 것:

- NPC `Behavior Parameters`: Vector Observation **120**, Discrete Branches **[9, 9]**, Behavior Name `TheThing`
- NPC `Decision Requester`: Decision Period **20**, Take Actions Between Decisions **켬**
- NPC 6명(Tag `NPC`), NavMeshSurface, 방 8개에 `InteractionPoint`(roomIndex 0~7), GameManager의 `cafeSpawnPoint`
- 학습 중(Python 연결 시) Player는 자동으로 제외된다(`includePlayerInTraining`)

## 학습

```bash
pip install torch numpy pyyaml tensorboard mlagents-envs
python mappo_trainer.py --env-path <빌드 경로>
```

- `scripts/train.sh`(mlagents-learn / POCA)는 이 구조와 맞지 않으니 쓰지 않는다.
- 트레이너 기본값: `entropy_coef 0.01`, 커리큘럼 없음. 비교 결과 이 값이 가장 나았다(`sim/results/interrupt_vs_entropy`).

## 검증 상태와 한계

- **컴파일**: Unity API 스텁으로 C# 9 컴파일을 통과했다(`tools/csharp_check`). Unity 에디터에서는 아직 열어 보지 않았다.
- **확인 필요**
  - 분산 스폰 위치가 실제 맵에서 걸을 수 있는 곳인지 (`NavMesh.SamplePosition` 보정)
  - 자동 수리, 이동 확정, 근거 사격이 의도대로 동작하는지 (한 판 관전으로 확인)
  - 짧은 학습(약 20만 스텝)에서 에피소드 수 = 게임 수인지, 게임당 부수기·수리가 시뮬레이터와 비슷한지
- **결과의 한계**: 시뮬레이터는 NavMesh 대신 직선 이동, 규칙봇은 단순화 버전이다. 인간팀과 함장 정책은 4M에서도 무작위에 가깝고, 대응의 상당 부분은 규칙(자동 수리, 근거 사격)이 맡는다.
