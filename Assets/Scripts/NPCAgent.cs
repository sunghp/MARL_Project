using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

/// <summary>
/// MAPPO 에이전트 (관측 120 = Actor 64 + Critic 56, 2 브랜치 행동)
///
/// 행동:
///   Branch 0 (이산 9): 방 선택 0~7, 대기 8
///   Branch 1 (이산 9): 없음(0), 부수기(1), 고치기(2), 사격 슬롯0~5(3~8)
///
/// 캐릭터 슬롯: GameManager.GetCharacterSlots()(에피소드 내내 고정된 순서)에서 자기 자신을 뺀 순서.
///   관측의 "다른 캐릭터 k번"과 사격 "슬롯 k"가 같은 캐릭터를 가리킨다.
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
    public float winReward = 5f;                  // ±10→±5: 과정 보상이 승패에 묻히지 않게 (구멍2)
    public float loseReward = -5f;
    public float sabotageRewardHidden = 1.0f;     // 목격 없이 부수기 (은닉 성공) — 핵심 정렬 신호
    public float sabotageRewardWitnessed = 0.0f;  // 목격당하며 부수기: 보너스 없음(데미지 스텝보상은 그대로 받음)
    public float repairReward = 0.5f;
    public float healthChangeRewardScale = 0.01f;

    [Header("정규화 설정")]
    public float mapSize = 50f;

    // 내부 상태
    private float previousAverageHealth;
    private int prevAlertRoom = -1;       // 알림 방 접근 셰이핑: 직전에 가장 가까웠던 알림 방
    private float prevAlertDist = 0f;
    private bool isSaboteur;
    private bool isCaptain;
    private InteractionPoint[] allRooms;

    // ===== 상수 =====
    private const int MAX_ROOMS = 8;
    private const int MAX_OTHER_CHARACTERS = 6;   // 7명 이하 게임 가정 (현재 NPC 6명)
    private const int SHOOT_ACTION_OFFSET = 3;      // Branch 1의 3~8 = 슬롯 0~5 사격

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
        prevAlertRoom = -1;
    }

    // ===================================================================
    // Observations (120차원 = LOCAL 64 + GLOBAL 56)
    // ===================================================================
    //
    // LOCAL (Actor, 부분관측):
    //   자기 정보 5 (역할2 + 위치3) + 공개 게임 상태 3
    //   방 8 × 4 (거리, 안정도, 사용중, 시야플래그)                  = 32
    //   다른 캐릭터 6 × 4 (x, z, 시야플래그, 부수는 걸 직접 목격함)   = 24
    //   합계 64
    //
    // GLOBAL (Critic, 전역 — CTDE이므로 실제 역할까지 포함):
    //   자기 정보 5 + 게임 상태 3
    //   방 8 × 3 (거리, 안정도, 사용중)                              = 24
    //   다른 캐릭터 6 × 4 (x, z, 생존, 사보타주 여부)                = 24
    //   합계 56
    //
    // ===================================================================
    private const int LOCAL_OBS_DIM = 64;
    private const int GLOBAL_OBS_DIM = 56;

    // 고정 슬롯 순서로 k번째 "다른 캐릭터" (없으면 null)
    GameObject GetOtherSlot(int k)
    {
        if (gameManager == null) return null;
        var slots = gameManager.GetCharacterSlots();
        int idx = 0;
        foreach (var ch in slots)
        {
            if (ch == gameObject) continue;
            if (idx == k) return ch;
            idx++;
        }
        return null;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (roleManager == null) roleManager = RoleManager.Instance;
        if (gameManager == null) gameManager = GameManager.Instance;
        if (roleManager != null) {
            isSaboteur = roleManager.IsSaboteur(gameObject);
            isCaptain  = roleManager.IsCaptain(gameObject);
        }

        float visionRange     = gameManager != null ? gameManager.visionRange : 10f;
        float alertThreshold  = gameManager != null ? gameManager.locationAlertThreshold : 30f;

        float avgHealth    = SystemHealth.Instance != null ? SystemHealth.Instance.GetAverageHealth() : 0f;
        float distProgress = gameManager != null ? gameManager.GetDistanceProgress() : 0f;
        float aliveRatio   = gameManager != null ? gameManager.GetAliveHumanRatio() : 0f;

        // ===================== LOCAL (Actor, 64) =====================
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

        // 다른 캐릭터 6명 (6 × 4 = 24): 고정 슬롯 순서. 위치는 시야 안일 때만,
        // 목격 플래그는 "내가 이 캐릭터가 부수는 걸 직접 본 적 있음" (추리 근거)
        for (int k = 0; k < MAX_OTHER_CHARACTERS; k++)
        {
            GameObject ch = GetOtherSlot(k);
            bool alive = ch != null && ch.activeInHierarchy;
            bool visible = alive &&
                Vector3.Distance(transform.position, ch.transform.position) <= visionRange;

            sensor.AddObservation(visible ? ch.transform.position.x / mapSize : 0f);
            sensor.AddObservation(visible ? ch.transform.position.z / mapSize : 0f);
            sensor.AddObservation(visible ? 1f : 0f);
            sensor.AddObservation(alive && gameManager.HasWitnessedSabotage(gameObject, ch) ? 1f : 0f);
        }

        // ===================== GLOBAL (Critic, 56) =====================
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

        // 모든 캐릭터 (6 × 4 = 24): 가림 없음, 실제 역할 포함 (Critic 전용)
        for (int k = 0; k < MAX_OTHER_CHARACTERS; k++)
        {
            GameObject ch = GetOtherSlot(k);
            bool alive = ch != null && ch.activeInHierarchy;
            sensor.AddObservation(alive ? ch.transform.position.x / mapSize : 0f);
            sensor.AddObservation(alive ? ch.transform.position.z / mapSize : 0f);
            sensor.AddObservation(alive ? 1f : 0f);
            sensor.AddObservation(alive && roleManager != null && roleManager.IsSaboteur(ch) ? 1f : 0f);
        }
        // 합계: 64 + 56 = 120
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
        int interactionChoice = actions.DiscreteActions[1]; // 0: 없음, 1: 부수기, 2: 고치기, 3~8: 슬롯 사격

        // ===== 사격 처리 (방 근처일 필요 없음, 대상이 시야 안이어야 함) =====
        if (interactionChoice >= SHOOT_ACTION_OFFSET)
        {
            if (isCaptain)
                TryShoot(interactionChoice - SHOOT_ACTION_OFFSET);
        }

        // ===== 이동 처리 =====
        // 상호작용 진행 중이면 이동 명령 무시 (중단 방지)
        if (!npcController.IsInteracting() &&
            roomChoice < allRooms.Length && allRooms[roomChoice] != null)
        {
            npcController.MoveToRoom(allRooms[roomChoice]);
        }
        // roomChoice == 8: 현재 위치 유지

        // ===== 상호작용 처리 =====
        if ((interactionChoice == 1 || interactionChoice == 2) &&
            !npcController.IsInteracting() && npcController.IsNearInteractionPoint())
        {
            if (interactionChoice == 1)
            {
                if (isSaboteur)
                    npcController.TryStartSabotage();
            }
            else
            {
                npcController.TryStartRepair();
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

        // 알림 방 접근 셰이핑 (인간팀): 가장 가까운 알림 방까지 거리가 줄어든 만큼 보상
        float k = gameManager != null ? gameManager.rewardAlertApproach : 0f;
        if (k != 0f && !isSaboteur)
        {
            int nearest = -1;
            float nearestDist = float.MaxValue;
            for (int i = 0; i < allRooms.Length; i++)
            {
                if (allRooms[i] == null || !allRooms[i].IsAlerted()) continue;
                float d = Vector3.Distance(transform.position, allRooms[i].transform.position);
                if (d < nearestDist) { nearestDist = d; nearest = i; }
            }
            if (nearest >= 0)
            {
                if (nearest == prevAlertRoom)
                    AddReward(k * (prevAlertDist - nearestDist));
                prevAlertRoom = nearest;
                prevAlertDist = nearestDist;
            }
            else
            {
                prevAlertRoom = -1;
            }
        }
    }

    // ===== 셰이핑 이벤트 (GameManager/InteractionPoint/CaptainGun에서 호출) =====
    bool ControlledByPolicy() => npcController != null && npcController.IsUsingML();

    public void OnWitnessSabotage()
    {
        if (!isSaboteur && ControlledByPolicy() && gameManager != null)
            AddReward(gameManager.rewardWitness);
    }

    public void OnInterruptSabotage()
    {
        if (!isSaboteur && ControlledByPolicy() && gameManager != null)
            AddReward(gameManager.rewardInterrupt);
    }

    public void OnCaptainHit()
    {
        if (isCaptain && ControlledByPolicy() && gameManager != null)
            AddReward(gameManager.rewardCaptainHit);
    }

    public void OnSabotageComplete(bool wasHidden)
    {
        if (isSaboteur)
            AddReward(wasHidden ? sabotageRewardHidden : sabotageRewardWitnessed);
    }

    // repairedFraction: 실제로 회복된 양 / repairAmount (0~1). 이미 만땅인 방 수리는 0 → 보상 없음
    public void OnRepairComplete(float repairedFraction)
    {
        if (!isSaboteur)
            AddReward(repairReward * Mathf.Clamp01(repairedFraction));
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

    void TryShoot(int slot)
    {
        var captainGun = GetComponent<CaptainGun>();
        if (captainGun == null || captainGun.GetRemainingBullets() <= 0) return;

        // 선택한 슬롯의 캐릭터: 유예 시간, 근거(목격 기록), 사거리 규칙을 통과해야 사격
        // (통과 못 하면 아무 일도 없음 → 근거 없는 무작위 오사 불가)
        GameObject target = GetOtherSlot(slot);
        if (!gameManager.CanShootTarget(gameObject, target)) return;

        captainGun.TryExecuteTarget(target);
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
        if (Input.GetKey(KeyCode.F)) discreteActions[1] = SHOOT_ACTION_OFFSET; // 슬롯 0 사격
    }
}