using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

/// <summary>
/// MAPPO 에이전트 (42차원 관측, 2 브랜치 행동)
///
/// 관측 (42):
///   자기 정보(5) + 게임 상태(3) + 방 상태(24) + 다른 캐릭터(10)
///
/// 행동:
///   Branch 0 (이산 9): 방 선택 0~7, 대기 8
///   Branch 1 (이산 4): 없음(0), 부수기(1), 고치기(2), 사격(3)
/// </summary>
public class NPCAgent : Agent
{
    // ===================================================================
    // 필드
    // ===================================================================

    [Header("참조")]
    private NPCController npcController;
    private GameManager gameManager;
    private RoleManager roleManager;

    [Header("보상 설정")]
    public float winReward = 10f;
    public float loseReward = -10f;
    public float sabotageReward = 0.5f;
    public float repairReward = 0.5f;
    public float healthChangeRewardScale = 0.01f;

    [Header("정규화 설정")]
    public float mapSize = 50f;

    // 내부 상태
    private float previousAverageHealth;
    private bool isSaboteur;
    private bool isCaptain;
    private InteractionPoint[] allRooms;

    // ===== 상수 =====
    private const int MAX_ROOMS = 8;
    private const int MAX_OTHER_CHARACTERS = 5;

    // ===================================================================
    // 초기화
    // ===================================================================

    public override void Initialize()
    {
        npcController = GetComponent<NPCController>();
        gameManager = GameManager.Instance;
        roleManager = RoleManager.Instance;

        // 방 목록을 roomIndex로 정렬
        allRooms = FindObjectsOfType<InteractionPoint>();
        System.Array.Sort(allRooms, (a, b) => a.roomIndex.CompareTo(b.roomIndex));

        Debug.Log($"[NPCAgent] {gameObject.name} 초기화 - 감지된 방 개수: {allRooms.Length}");
    }

    public override void OnEpisodeBegin()
    {
        if (roleManager == null) roleManager = RoleManager.Instance;
        if (gameManager == null) gameManager = GameManager.Instance;

        if (roleManager == null || gameManager == null)
        {
            Debug.LogWarning($"[NPCAgent] {gameObject.name} - Manager가 아직 초기화 안됨");
            return;
        }

        isSaboteur = roleManager.IsSaboteur(gameObject);
        isCaptain = roleManager.IsCaptain(gameObject);

        if (SystemHealth.Instance != null)
            previousAverageHealth = SystemHealth.Instance.GetAverageHealth();
    }

    // ===================================================================
    // Observations (42차원)
    // ===================================================================
    //
    // 자기 정보:      2(역할) + 3(위치) = 5
    // 게임 상태:      3
    // 방 상태:        8방 × 3(거리, 안정도, 사용중) = 24
    // 다른 캐릭터:    5명 × 2(x, z) = 10
    // 총합:           42
    //
    // ===================================================================
    private const int LOCAL_OBS_DIM = 55;
    private const int GLOBAL_OBS_DIM = 42;

    public override void CollectObservations(VectorSensor sensor)
    {
        if (roleManager == null) roleManager = RoleManager.Instance;
        if (roleManager != null) {
            isSaboteur = roleManager.IsSaboteur(gameObject);
            isCaptain  = roleManager.IsCaptain(gameObject);
        }

        float visionRange     = gameManager != null ? gameManager.visionRange : 10f;
        float alertThreshold  = gameManager != null ? gameManager.locationAlertThreshold : 30f;

        float avgHealth    = SystemHealth.Instance != null ? SystemHealth.Instance.GetAverageHealth() : 0f;
        float distProgress = gameManager != null ? gameManager.GetDistanceProgress() : 0f;
        float aliveRatio   = gameManager != null ? gameManager.GetAliveHumanRatio() : 0f;

        // ===================== LOCAL (Actor, 55) =====================
        // 자기 정보 (5)
        sensor.AddObservation(isSaboteur ? 1f : 0f);
        sensor.AddObservation(isCaptain ? 1f : 0f);
        sensor.AddObservation(transform.position / mapSize);   // 3

        // 공개 게임 상태 (3)
        sensor.AddObservation(avgHealth / 100f);
        sensor.AddObservation(distProgress);
        sensor.AddObservation(aliveRatio);

        // 방 8개 (8 × 4 = 32): 거리는 항상, 안정도/사용중은 (시야 OR 알림)일 때만
        for (int i = 0; i < MAX_ROOMS; i++)
        {
            if (i < allRooms.Length && allRooms[i] != null)
            {
                float dist = Vector3.Distance(transform.position, allRooms[i].transform.position);
                sensor.AddObservation(dist / mapSize);   // 거리 (공개)

                bool alerted = allRooms[i].GetCurrentHealth() <= alertThreshold;
                bool visible = (dist <= visionRange) || alerted;
                
                if (visible)
                {
                    sensor.AddObservation(allRooms[i].GetHealthPercent());        // 안정도
                    sensor.AddObservation(allRooms[i].IsBeingUsed() ? 1f : 0f);   // 사용중
                    sensor.AddObservation(1f);                                    // visFlag
                }
                else
                {
                    sensor.AddObservation(0f);   // 안정도 (가림)
                    sensor.AddObservation(0f);   // 사용중 (가림)
                    sensor.AddObservation(0f);   // visFlag = 안 보임
                }
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        // 캐릭터 5명 (5 × 3 = 15): 시야 안일 때만 위치 공개
        int cCount = 0;
        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var ch in gameManager.allCharacters)
            {
                if (ch == gameObject) continue;
                if (cCount >= MAX_OTHER_CHARACTERS) break;

                if (ch != null && ch.activeInHierarchy &&
                    Vector3.Distance(transform.position, ch.transform.position) <= visionRange)
                {
                    sensor.AddObservation(ch.transform.position.x / mapSize);
                    sensor.AddObservation(ch.transform.position.z / mapSize);
                    sensor.AddObservation(1f);   // visFlag
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
                cCount++;
            }
        }
        for (int i = cCount; i < MAX_OTHER_CHARACTERS; i++)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // ===================== GLOBAL (Critic, 42) =====================
        // 자기 정보 (5)
        sensor.AddObservation(isSaboteur ? 1f : 0f);
        sensor.AddObservation(isCaptain ? 1f : 0f);
        sensor.AddObservation(transform.position / mapSize);

        // 게임 상태 (3)
        sensor.AddObservation(avgHealth / 100f);
        sensor.AddObservation(distProgress);
        sensor.AddObservation(aliveRatio);

        // 모든 방 (8 × 3 = 24): 가림 없음
        for (int i = 0; i < MAX_ROOMS; i++)
        {
            if (i < allRooms.Length && allRooms[i] != null)
            {
                float dist = Vector3.Distance(transform.position, allRooms[i].transform.position);
                sensor.AddObservation(dist / mapSize);
                sensor.AddObservation(allRooms[i].GetHealthPercent());
                sensor.AddObservation(allRooms[i].IsBeingUsed() ? 1f : 0f);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        // 모든 캐릭터 (5 × 2 = 10): 가림 없음
        int gCount = 0;
        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var ch in gameManager.allCharacters)
            {
                if (ch == gameObject) continue;
                if (gCount >= MAX_OTHER_CHARACTERS) break;
                if (ch != null && ch.activeInHierarchy)
                {
                    sensor.AddObservation(ch.transform.position.x / mapSize);
                    sensor.AddObservation(ch.transform.position.z / mapSize);
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
                gCount++;
            }
        }
        for (int i = gCount; i < MAX_OTHER_CHARACTERS; i++)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
        // 합계: 55 + 42 = 97
    }

    // ===================================================================
    // Actions
    // ===================================================================

    public override void OnActionReceived(ActionBuffers actions)
    {

        if (gameManager == null) gameManager = GameManager.Instance;
        if (npcController == null) npcController = GetComponent<NPCController>();

        if (npcController == null || gameManager == null || gameManager.IsGameOver()) return;

        // 평가 모드에서 규칙봇으로 고정된 캐릭터는 ML 행동을 무시 (NPCAIBrain이 제어)
        if (!npcController.IsUsingML()) return;

        int roomChoice = actions.DiscreteActions[0];        // 0~7: 방 선택, 8: 대기
        int interactionChoice = actions.DiscreteActions[1]; // 0: 없음, 1: 부수기, 2: 고치기, 3: 사격

        // ===== 이동 처리 =====
        // 상호작용 진행 중이면 이동 명령 무시 (중단 방지)
        if (!npcController.IsInteracting() &&
            roomChoice < allRooms.Length && allRooms[roomChoice] != null)
        {
            npcController.MoveToRoom(allRooms[roomChoice]);
        }
        // roomChoice == 8: 현재 위치 유지

        // ===== 상호작용 처리 =====
        if (interactionChoice > 0 && !npcController.IsInteracting() && npcController.IsNearInteractionPoint())
        {
            switch (interactionChoice)
            {
                case 1: // 부수기
                    if (isSaboteur)
                        npcController.TryStartSabotage();
                    break;
                case 2: // 고치기
                    npcController.TryStartRepair();
                    break;
                case 3: // 사격
                    if (isCaptain)
                        TryShoot();
                    break;
            }
        }

        // ===== 주기적 보상 =====
        CalculateStepReward();
    }

    // ===================================================================
    // 보상 함수
    // ===================================================================

    void CalculateStepReward()
    {
        if (SystemHealth.Instance == null) return;

        float currentHealth = SystemHealth.Instance.GetAverageHealth();
        float healthDelta = currentHealth - previousAverageHealth;

        if (isSaboteur)
            AddReward(-healthDelta * healthChangeRewardScale);
        else
            AddReward(healthDelta * healthChangeRewardScale);

        previousAverageHealth = currentHealth;
    }

    public void OnSabotageComplete()
    {
        if (isSaboteur)
            AddReward(sabotageReward);
    }

    public void OnRepairComplete()
    {
        if (!isSaboteur)
            AddReward(repairReward);
    }

    public void OnGameEnd(bool humanWin)
    {
        if (isSaboteur)
            AddReward(humanWin ? loseReward : winReward);
        else
            AddReward(humanWin ? winReward : loseReward);
    }

    public void OnDeath()
    {
        AddReward(-1f);
    }

    // ===================================================================
    // 함장 사격
    // ===================================================================

    void TryShoot()
    {
        var captainGun = GetComponent<CaptainGun>();
        if (captainGun == null || captainGun.GetRemainingBullets() <= 0) return;

        // 가장 가까운 캐릭터 타겟팅
        GameObject bestTarget = null;
        float bestDist = float.MaxValue;

        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var character in gameManager.allCharacters)
            {
                if (character == gameObject || character == null || !character.activeInHierarchy)
                    continue;

                float dist = Vector3.Distance(transform.position, character.transform.position);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestTarget = character;
                }
            }
        }

        if (bestTarget != null)
        {
            captainGun.TryExecuteTarget(bestTarget);
        }
    }

    // ===================================================================
    // Heuristic (수동 테스트용)
    // ===================================================================

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = 8; // 기본: 대기
        discreteActions[1] = 0; // 기본: 없음

        for (int i = 0; i < 8; i++)
        {
            if (Input.GetKey(KeyCode.Alpha1 + i))
            {
                discreteActions[0] = i;
                break;
            }
        }

        if (Input.GetKey(KeyCode.E)) discreteActions[1] = 2;
        if (Input.GetKey(KeyCode.Q)) discreteActions[1] = 1;
        if (Input.GetKey(KeyCode.F)) discreteActions[1] = 3;
    }
}