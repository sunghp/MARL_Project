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

    public override void CollectObservations(VectorSensor sensor)
    {
        // ===== 1. 자기 정보 (5) =====
        sensor.AddObservation(isSaboteur ? 1f : 0f);           // 1
        sensor.AddObservation(isCaptain ? 1f : 0f);            // 1
        sensor.AddObservation(transform.position / mapSize);    // 3 (x, y, z)

        // ===== 2. 게임 상태 (3) =====
        float avgHealth = 0f;
        if (SystemHealth.Instance != null)
            avgHealth = SystemHealth.Instance.GetAverageHealth();

        sensor.AddObservation(avgHealth / 100f);                                // 1
        sensor.AddObservation(gameManager != null ?
            gameManager.GetDistanceProgress() : 0f);                            // 1
        sensor.AddObservation(gameManager != null ?
            gameManager.GetAliveHumanRatio() : 0f);                             // 1

        // ===== 3. 각 방 상태 (MAX_ROOMS * 3 = 24) =====
        for (int i = 0; i < MAX_ROOMS; i++)
        {
            if (i < allRooms.Length && allRooms[i] != null)
            {
                float distance = Vector3.Distance(transform.position, allRooms[i].transform.position);
                sensor.AddObservation(distance / mapSize);                      // 거리 정규화
                sensor.AddObservation(allRooms[i].GetHealthPercent());          // 방 안정도 (0~1)
                sensor.AddObservation(allRooms[i].IsBeingUsed() ? 1f : 0f);    // 사용 중 여부
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        // ===== 4. 다른 캐릭터 정보 (MAX_OTHER_CHARACTERS * 2 = 10) =====
        int characterCount = 0;

        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var character in gameManager.allCharacters)
            {
                if (character == gameObject) continue;
                if (characterCount >= MAX_OTHER_CHARACTERS) break;

                if (character != null && character.activeInHierarchy)
                {
                    sensor.AddObservation(character.transform.position.x / mapSize);
                    sensor.AddObservation(character.transform.position.z / mapSize);
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
                characterCount++;
            }
        }

        // 남은 슬롯 패딩
        for (int i = characterCount; i < MAX_OTHER_CHARACTERS; i++)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // 총 Observation 수: 5 + 3 + 24 + 10 = 42
    }

    // ===================================================================
    // Actions
    // ===================================================================

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (gameManager == null || gameManager.IsGameOver()) return;

        int roomChoice = actions.DiscreteActions[0];        // 0~7: 방 선택, 8: 대기
        int interactionChoice = actions.DiscreteActions[1]; // 0: 없음, 1: 부수기, 2: 고치기, 3: 사격

        // ===== 이동 처리 =====
        if (roomChoice < allRooms.Length && allRooms[roomChoice] != null)
        {
            npcController.MoveToRoom(allRooms[roomChoice]);
        }
        // roomChoice == 8: 현재 위치 유지

        // ===== 상호작용 처리 =====
        if (interactionChoice > 0 && npcController != null && npcController.IsNearInteractionPoint())
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

        // 숫자키 1~8로 방 선택
        for (int i = 0; i < 8; i++)
        {
            if (Input.GetKey(KeyCode.Alpha1 + i))
            {
                discreteActions[0] = i;
                break;
            }
        }

        if (Input.GetKey(KeyCode.E)) discreteActions[1] = 2; // 고치기
        if (Input.GetKey(KeyCode.Q)) discreteActions[1] = 1; // 부수기
        if (Input.GetKey(KeyCode.F)) discreteActions[1] = 3; // 사격
    }
}