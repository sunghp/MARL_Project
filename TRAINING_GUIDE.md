# Unity 빌드에서 학습하기 (단계별)

- 트레이너는 `mappo_trainer.py`(커스텀 MAPPO)다. **`mlagents-learn`은 쓰지 않는다.**
- 학습 중에는 Python이 행동을 계산하고, Unity 빌드는 게임만 돌린다.
- **현재 방향: 인간팀(선원·함장)은 강화 규칙봇으로 고정하고, 사보타주 2명만 학습한다.**
  - `config/TheThing.yaml`의 `bot_team: 2`, `bot_level: 1`이 이 설정이다. 트레이너가 yaml을 읽고 인간팀 데이터는 모으지 않는다.
  - 근거와 결과: `sim/results/saboteur_vs_bot/REPORT.md`
  - 이전 방향(두 팀 공동 학습)은 `--balance-yaml config/TheThing_selfplay_v4.yaml`로 그대로 돌릴 수 있다.

---

## 0. 준비물

| 항목 | 버전 / 위치 |
|---|---|
| Unity | **2022.3.62f3** (`ProjectSettings/ProjectVersion.txt`) |
| ML-Agents Unity 패키지 | `com.unity.ml-agents` 2.0.2 (이미 `Packages/manifest.json`에 있음) |
| Python | 3.9 ~ 3.11 (3.10 권장) |
| 코드 | 브랜치 `claude/compassionate-bell-nwurht` |

## 1. Python 환경 만들기

```bash
conda create -n marl python=3.10
conda activate marl
pip install -r requirements.txt
```

- `requirements.txt`가 `mlagents-envs==0.28.0`, `numpy==1.23.5`, `protobuf==3.20.3`을 고정한다.
- 이 셋은 바꾸면 안 된다.
  - numpy 1.24 이상: `np.bool`이 없어 오류가 난다.
  - protobuf 4 이상: `Descriptors cannot be created directly` 오류가 난다.
- GPU를 쓰려면 torch를 CUDA 버전으로 따로 설치한다(예: `pip install torch --index-url https://download.pytorch.org/whl/cu121`). 네트워크가 작아서 CPU로도 충분하다.

## 2. Unity에서 프로젝트 열고 확인

1. Unity Hub에서 프로젝트를 열고, **Console에 컴파일 오류가 없는지** 확인한다.
2. `Assets/Scenes/SampleScene`을 연다.
3. 씬의 NPC 하나를 선택해 Inspector를 확인한다.
   - **Behavior Parameters**
     - Behavior Name `TheThing`
     - Vector Observation Space Size **120**
     - Discrete Branches **2개, 크기 9 / 9**
     - **Behavior Type = Default**. 프리팹 원본은 Heuristic Only지만, 씬의 NPC 6명은 Default로 덮어써져 있다. Heuristic Only면 학습이 안 된다.
     - Model = None
   - **Decision Requester**: Decision Period **20**, Take Actions Between Decisions **체크**
4. `GameManager` 오브젝트의 Inspector 값이 학습 설정과 같은지 확인한다.
   - Vision Range 10, Location Alert Threshold 60, Sabotage Damage 8, Shoot Range 15, Move Speed 5
   - Bot Level 1, Bot Perception 1, Bot Reaction 0, Bot Chase 체크
   - Commit Move / Auto Repair / Spawn Spread / Ship Speed By Health / Share Witness / Shoot Requires Evidence 체크
   - 학습 때는 yaml 값이 덮어쓰지만, 첫 판과 수동 플레이는 이 값을 쓴다.

## 3. 에디터로 짧게 시험 (빌드 전에 권장)

```bash
python mappo_trainer.py --editor --timesteps 20000
```

1. 위 명령을 실행하고 Python 창에 연결 대기가 뜨면, **Unity 에디터에서 Play**를 누른다.
2. 확인할 것
   - Python 창
     - `[환경] Behavior names: ['TheThing?team=0']`
     - `관측 shape: ... (120,)`
     - 오류 없이 `[에피소드 N]` 로그가 이어지는지
   - Unity Console
     - `[학습 모드] Player 제외`
     - NPC들이 **카페가 아니라 방 근처에서 시작**하는지(분산 스폰)
     - 방으로 이동하고, 부서진 방 근처의 인간팀이 자동으로 수리하는지
3. 에디터는 느리니까 여기서는 동작 확인만 하고 끝낸다.

## 4. 빌드 만들기

1. `File > Build Settings`
   - Platform: Windows(x86_64). Linux 서버에서 돌릴 거면 Linux.
   - Scenes In Build: `Scenes/SampleScene`이 체크돼 있는지 확인(이미 등록돼 있음).
2. `Player Settings > Resolution and Presentation`
   - Run In Background: **켬**(이미 켜져 있음). 꺼져 있으면 창이 뒤로 가는 순간 학습이 멈춘다.
   - Fullscreen Mode: **Windowed**로 바꾸는 것을 권장한다. 학습은 화면 없이 돌지만, 나중에 관전(`spectate.py`)할 때 창으로 뜨는 편이 편하다.
3. **Build**: 프로젝트 폴더 아래 `Builds/Windows/`에 `My project.exe` 이름으로 저장한다.
   - 트레이너의 기본 경로가 `Builds/Windows/My project.exe`라서, 이름을 맞추면 `--env-path`를 생략할 수 있다.

## 5. 빌드로 짧은 학습 (시뮬레이터와 비교)

```bash
python mappo_trainer.py --env-path "Builds/Windows/My project.exe" --timesteps 200000 --entropy-role saboteur=0.003
```

- 화면 없이(`-nographics`), 게임 속도 20배로 돈다. 화면을 보며 디버그하려면 `--graphics`를 붙인다.
- **학습 속도를 잰다.** 콘솔의 `스텝:` 값이 늘어나는 속도로 초당 스텝을 구하고, 본 학습 시간을 예상한다(4M ÷ 초당 스텝).
- 콘솔에 `[고정 봇] 인간팀은 항상 규칙봇 → 그 팀은 학습하지 않음`이 떠야 한다.
- 10만 스텝마다 규칙봇 상대 평가(사보타주만)가 자동으로 돌고, 평가 스텝은 학습 스텝에 포함되지 않는다.
- 아래가 시뮬레이터와 비슷하면 정상이다(같은 설정의 시뮬레이터 0~0.25M 구간, 시드 2개).

| TensorBoard 지표 | 시뮬레이터 0~0.25M |
|---|---|
| `outcome/saboteur_win_rate_vs_bot` | 0 (인간팀 전승) |
| `behavior/sabotage_count` | 게임당 1 → 4 |
| `behavior/repair_count` | 게임당 1~3 |
| `episode/seconds`(실제 시간)와 게임 수 | 에피소드 1개 = 게임 1판 (게임 하나가 여러 에피소드로 쪼개지지 않음) |

- **규칙봇끼리 기준선도 함께 잰다.** 학습 없이 에디터에서 Play만 누르면 된다.
  - GameManager Inspector에서 Eval Mode를 켜고 Frozen Team을 3으로 둔다. 그러면 전원이 규칙봇이 된다.
  - 시뮬레이터 값: 규칙봇 사보타주 vs 강화 봇 인간팀은 사보타주 승률 42%, 판당 부수기 24회, 처형된 사보타주 1.3명.
  - Console의 게임 종료 로그로 몇십 판을 세어 비슷한지 본다. 크게 다르면 C# 이식(NavMesh 이동 포함)부터 확인한다.

- 크게 다르면 본 학습 전에 원인부터 본다. 예: 게임이 너무 빨리 끝남, 부수기가 전혀 늘지 않음, NPC가 방에 도착하지 못함.

## 6. 본 학습

```bash
python mappo_trainer.py --env-path "Builds/Windows/My project.exe" --timesteps 8000000 --entropy-role saboteur=0.003
```

- 시뮬레이터에서는 8M 스텝에서 시드에 따라 15~41%가 나왔다. 시드 2개 이상을 돌리고 좋은 쪽을 고른다.
- `./scripts/train.sh`는 엔트로피 옵션을 넘기지 않는다. 위 명령을 직접 쓴다.
- 실험용 옵션 (기본 꺼짐, 효과는 `sim/results/saboteur_improve/REPORT.md`)
  - `--action-mask`: 역할별로 효과 없는 상호작용을 막음
  - `--min-update-samples 4096`: 여러 판을 모아 한 번에 업데이트. 시드 편차가 가장 작았다
  - yaml `sab_obs: 1` / `reward_sab_witnessed: -1`: 사보타주 관측 v2 / 목격 페널티. 이득이 확인되지 않았다
    - 관측을 바꾸면 이전 모델과 호환되지 않는다. 관측과 체크포인트를 같은 설정으로 맞춘다

- 결과 위치
  - 체크포인트: `checkpoints/mappo_ep<에피소드>.pt` (50 에피소드마다)
  - TensorBoard 로그: `runs/mappo/`
- **중단과 재개**
  - `Ctrl+C`로 멈추면 `checkpoints/mappo_interrupted.pt`가 저장된다.
  - 이어서 하려면 `python mappo_trainer.py --load checkpoints/mappo_interrupted.pt --env-path "..." --timesteps 4000000` (총 스텝은 누적 기준).
- 같은 포트(5004)를 쓰므로 **학습은 한 번에 하나만** 돌린다.

## 7. 모니터링

```bash
tensorboard --logdir runs/mappo      # 또는 ./scripts/monitor.sh
```

브라우저에서 `http://localhost:6006`을 연다. 볼 지표와 시뮬레이터 기준값(같은 설정 `--entropy-role saboteur=0.003`, 시드 2개):

| 지표 | 의미 | 2M | 4M | 8M |
|---|---|---|---|---|
| `outcome/saboteur_win_rate_vs_bot` | 사보타주 승률 (규칙봇 인간팀 상대) | 0.13~0.14 | 0.05~0.30 | 0.15~0.37 |
| `eval/saboteur_vs_bot_win` | 같은 것, 평가 모드에서 30판 | 위와 비슷 | | |
| `behavior/sabotage_count` | 게임당 부수기 | 13~15 | 14~17 | 16~19 |
| `behavior/repair_count` | 게임당 인간팀 수리 | 5~6 | 5.5~6 | 6 |
| `behavior/sabotage_interrupted` | 게임당 끊기 | 1~2 | 1~1.5 | 1~1.6 |
| `policy/entropy_saboteur` | 사보타주 정책 무작위성 (최대 4.39) | 1~2 | 1~2 | 1~2 |

- 비교 기준(규칙봇 인간팀 상대, 학습 없음): 규칙봇 사보타주 0.42, 방 붙기 스크립트 0.17, 무작위 0
- 인간팀 쪽 지표(`policy/entropy_human` 등)는 학습하지 않으므로 기록되지 않는다.

## 8. 학습 결과 관전

```bash
# 빌드로 (창 모드 빌드 권장)
python spectate.py --checkpoint checkpoints/mappo_ep5000.pt --env-path "Builds/Windows/My project.exe"

# 에디터로: 실행 후 에디터에서 Play
python spectate.py --checkpoint checkpoints/mappo_ep5000.pt --editor
```

- 정상 속도(`--time-scale 1.0`)에, 학습 때와 같은 확률적 행동으로 돈다.
- `--episodes N`으로 판 수를 정하고, `--deterministic`을 주면 argmax 행동을 쓴다.
- 시뮬레이터에서 학습한 `models/saboteur_vs_bot_8M.pt`도 같은 방법으로 볼 수 있다(Unity에서 학습한 모델은 아님). 인간팀은 규칙봇이 움직인다.

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `The Unity environment took too long to respond` | 빌드 경로가 틀림, 방화벽이 5004 포트를 막음, 다른 학습이나 에디터가 이미 5004를 씀 |
| `AttributeError: module 'numpy' has no attribute 'bool'` | numpy 1.24 이상이 설치됨 → `pip install numpy==1.23.5` |
| `TypeError: Descriptors cannot be created directly` | protobuf 4 이상이 설치됨 → `pip install protobuf==3.20.3` |
| 관측 크기나 행동 크기 불일치 오류 | NPC Behavior Parameters가 120 / [9, 9]가 아님 |
| NPC가 전혀 안 움직이고 학습도 진행 안 됨 | Behavior Type이 Heuristic Only로 되어 있음 → Default로 |
| 창을 내리면 학습이 멈춤 | Player Settings의 Run In Background가 꺼짐 |
| 시작하자마자 NPC가 바닥 밖이나 이상한 위치에 있음 | 분산 스폰 위치가 NavMesh 밖 → 방 근처 NavMesh 확인, 또는 GameManager의 Spawn Spread를 끄고 비교 |
