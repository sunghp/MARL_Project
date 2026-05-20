using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;
using System.Collections.Generic;

/// <summary>
/// 계층적 MAPPO 에이전트
///
/// 고수준 정책 (이 스크립트가 학습):
///   - 목적지 선택, 상호작용 결정, 긴급 회피
///   - 사회적 정보(감시자, 시야) 기반 전략적 판단
///
/// 저수준 실행 (NavMesh가 처리):
///   - 경로 탐색 및 물리적 이동
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
    public float stealthBonus = 0.5f;
    public float healthChangeRewardScale = 0.01f;

    [Header("정규화 설정")]
    public float mapSize = 50f;

    [Header("시야 설정")]
    public float visionRange = 15f;
    public LayerMask obstacleLayer;

    // 내부 상태
    private float previousAverageHealth;
    private bool isSaboteur;
    private bool isCaptain;
    private InteractionPoint[] allRooms;

    // ===== 상수 =====
    private const int MAX_ROOMS = 8;
    private const int MAX_OTHER_CHARACTERS = 5;

    // ===================================================================
    // 관측 구조 (72차원)
    // ===================================================================
    //
    // [자기 상태]     7  역할(2) + 위치(3) + 속도(2)
    // [현재 행동]     3  상호작용중(1) + 진행률(1) + 부수기여부(1)
    // [사회적 정보]   6  시야내(1) + 최근접(1) + 감시자(1) + 감시자거리(1) + 동료(1) + 함장거리(1)
    // [게임 상태]     4  평균안정도(1) + 항해진행(1) + 인간비율(1) + 사보타주비율(1)
    // [방 상태]      32  8방 x 4(거리, 안정도, 사용중, 방향각)
    // [다른 캐릭터]  20  5명 x 4(위치xz, 거리, 상호작용중)
    // 총합: 72
    //
    // ===================================================================
    // 행동 구조 (3 브랜치)
    // ===================================================================
    //
    // Branch 0 (이산 10): 목적지 — 방0~7, 유지(8), 카페(9)
    // Branch 1 (이산 5):  상호작용 — 없음(0), 부수기(1), 고치기(2), 중단(3), 사격(4)
    // Branch 2 (이산 2):  긴급회피 — 정상(0), 이탈(1)

    // ===================================================================
    // 초기화
    // ===================================================================

    public override void Initialize()
    {
        npcController = GetComponent<NPCController>();
        gameManager = GameManager.Instance;
        roleManager = RoleManager.Instance;

        // 방 목록을 roomIndex로 정렬 → 관측 순서 안정화
        allRooms = FindObjectsOfType<InteractionPoint>();
        System.Array.Sort(allRooms, (a, b) => a.roomIndex.CompareTo(b.roomIndex));

        if (GameManager.Instance != null)
            visionRange = GameManager.Instance.visionRange;

        Debug.Log($"[NPCAgent] {gameObject.name} 초기화 - 방 {allRooms.Length}개 (정렬됨)");
    }

    public override void OnEpisodeBegin()
    {
        if (roleManager == null) roleManager = RoleManager.Instance;
        if (gameManager == null) gameManager = GameManager.Instance;

        if (roleManager == null || gameManager == null)
        {
            Debug.LogWarning($"[NPCAgent] {gameObject.name} - Manager 미초기화");
            return;
        }

        isSaboteur = roleManager.IsSaboteur(gameObject);
        isCaptain = roleManager.IsCaptain(gameObject);

        // TeamId 동적 설정: 인간팀(0), 사보타주팀(1)
        var behaviorParams = GetComponent<BehaviorParameters>();
        if (behaviorParams != null)
            behaviorParams.TeamId = isSaboteur ? 1 : 0;

        if (SystemHealth.Instance != null)
            previousAverageHealth = SystemHealth.Instance.GetAverageHealth();
    }

    // ===================================================================
    // Observations (72차원)
    // ===================================================================

    public override void CollectObservations(VectorSensor sensor)
    {
        // ===== 1. 자기 상태 (7) =====
        sensor.AddObservation(isSaboteur ? 1f : 0f);                       // 1
        sensor.AddObservation(isCaptain ? 1f : 0f);                        // 1
        sensor.AddObservation(transform.position / mapSize);                // 3

        var navAgent = GetComponent<UnityEngine.AI.NavMeshAgent>();
        Vector3 vel = (navAgent != null && navAgent.enabled) ? navAgent.velocity : Vector3.zero;
        float maxSpeed = gameManager != null ? gameManager.moveSpeed : 5f;
        maxSpeed = Mathf.Max(maxSpeed, 0.1f);
        sensor.AddObservation(vel.x / maxSpeed);                            // 1
        sensor.AddObservation(vel.z / maxSpeed);                            // 1

        // ===== 2. 현재 행동 상태 (3) =====
        bool interacting = npcController != null && npcController.IsInteracting();
        sensor.AddObservation(interacting ? 1f : 0f);                      // 1
        sensor.AddObservation(npcController != null ?
            npcController.GetInteractionProgress() : 0f);                  // 1
        sensor.AddObservation(npcController != null &&
            npcController.isSabotaging ? 1f : 0f);                        // 1

        // ===== 3. 사회적 정보 (6) =====
        CollectSocialObservations(sensor);

        // ===== 4. 게임 상태 (4) =====
        float avgHealth = SystemHealth.Instance != null ?
            SystemHealth.Instance.GetAverageHealth() : 100f;
        sensor.AddObservation(avgHealth / 100f);                            // 1
        sensor.AddObservation(gameManager != null ?
            gameManager.GetDistanceProgress() : 0f);                       // 1
        sensor.AddObservation(gameManager != null ?
            gameManager.GetAliveHumanRatio() : 0f);                        // 1

        int sabAlive = gameManager != null ? gameManager.GetAliveSaboteurCount() : 0;
        int sabTotal = gameManager != null ? gameManager.sabotageCount : 2;
        sensor.AddObservation(sabTotal > 0 ? (float)sabAlive / sabTotal : 0f); // 1

        // ===== 5. 방 상태 (MAX_ROOMS * 4 = 32) =====
        for (int i = 0; i < MAX_ROOMS; i++)
        {
            if (i < allRooms.Length && allRooms[i] != null)
            {
                Vector3 toRoom = allRooms[i].transform.position - transform.position;
                float dist = toRoom.magnitude;

                sensor.AddObservation(dist / mapSize);                      // 거리
                sensor.AddObservation(allRooms[i].GetHealthPercent());      // 안정도 (0~1)
                sensor.AddObservation(allRooms[i].IsBeingUsed() ? 1f : 0f); // 사용 중

                // 내 정면 기준 방향각 (-1 ~ 1)
                float angle = dist > 0.01f ?
                    Vector3.SignedAngle(transform.forward, toRoom.normalized, Vector3.up) : 0f;
                sensor.AddObservation(angle / 180f);                        // 방향
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        // ===== 6. 다른 캐릭터 (MAX_OTHER_CHARACTERS * 4 = 20) =====
        int charCount = 0;
        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var character in gameManager.allCharacters)
            {
                if (character == gameObject) continue;
                if (charCount >= MAX_OTHER_CHARACTERS) break;

                if (character != null && character.activeInHierarchy)
                {
                    float dist = Vector3.Distance(transform.position, character.transform.position);
                    sensor.AddObservation(character.transform.position.x / mapSize); // x
                    sensor.AddObservation(character.transform.position.z / mapSize); // z
                    sensor.AddObservation(dist / mapSize);                            // 거리

                    NPCController otherNpc = character.GetComponent<NPCController>();
                    sensor.AddObservation(
                        otherNpc != null && otherNpc.IsInteracting() ? 1f : 0f);    // 상호작용
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
                charCount++;
            }
        }

        for (int i = charCount; i < MAX_OTHER_CHARACTERS; i++)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    /// <summary>
    /// 사회적 정보 수집 (6차원)
    /// - 시야 내 캐릭터 수 / 최근접 거리
    /// - 나를 감시하는 캐릭터 수 / 최근접 감시자 거리
    /// - 근처 사보타주 동료 수 (사보타주만)
    /// - 함장까지 거리
    /// </summary>
    void CollectSocialObservations(VectorSensor sensor)
    {
        int visibleCount = 0;
        float nearestDist = mapSize;
        int watcherCount = 0;
        float nearestWatcherDist = mapSize;
        int nearbyAllies = 0;
        float captainDist = mapSize;

        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var character in gameManager.allCharacters)
            {
                if (character == gameObject || character == null || !character.activeInHierarchy)
                    continue;

                float dist = Vector3.Distance(transform.position, character.transform.position);

                // 시야 내 캐릭터 (내가 볼 수 있는 모든 캐릭터)
                if (CanSeeTarget(character))
                {
                    visibleCount++;
                    if (dist < nearestDist) nearestDist = dist;

                    // 감시자 = 시야 내 인간팀 캐릭터 (사보타주에겐 위협, 인간에겐 아군)
                    bool isHumanTeam = roleManager != null && !roleManager.IsSaboteur(character);
                    if (isHumanTeam)
                    {
                        watcherCount++;
                        if (dist < nearestWatcherDist) nearestWatcherDist = dist;
                    }
                }

                // 사보타주 동료 (사보타주일 때만 유효)
                if (isSaboteur && roleManager != null &&
                    roleManager.IsSaboteur(character) && dist <= visionRange)
                    nearbyAllies++;

                // 함장 거리
                if (roleManager != null && roleManager.IsCaptain(character))
                    captainDist = dist;
            }
        }

        sensor.AddObservation(Mathf.Min(visibleCount, 5) / 5f);                    // 1
        sensor.AddObservation(nearestDist / mapSize);                               // 1
        sensor.AddObservation(Mathf.Min(watcherCount, 5) / 5f);                    // 1
        sensor.AddObservation(nearestWatcherDist / mapSize);                        // 1
        sensor.AddObservation(isSaboteur ? Mathf.Min(nearbyAllies, 2) / 2f : 0f); // 1
        sensor.AddObservation(captainDist / mapSize);                               // 1
    }

    // ===================================================================
    // Action Masking
    // ===================================================================

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // === Branch 0: 목적지 (0~7 방, 8 유지, 9 카페) ===
        for (int i = 0; i < 8; i++)
        {
            if (i >= allRooms.Length || allRooms[i] == null)
                actionMask.SetActionEnabled(0, i, false);
        }

        // === Branch 1: 상호작용 (0 없음, 1 부수기, 2 고치기, 3 중단, 4 사격) ===
        if (!isSaboteur)
            actionMask.SetActionEnabled(1, 1, false);       // 인간 → 부수기 불가

        if (!isCaptain)
            actionMask.SetActionEnabled(1, 4, false);       // 비함장 → 사격 불가

        bool interacting = npcController != null && npcController.IsInteracting();
        if (!interacting)
            actionMask.SetActionEnabled(1, 3, false);       // 비상호작용 → 중단 불가

        bool nearPoint = npcController != null && npcController.IsNearInteractionPoint();
        if (!nearPoint && !interacting)
        {
            actionMask.SetActionEnabled(1, 1, false);       // 방 근처 아님 → 부수기 불가
            actionMask.SetActionEnabled(1, 2, false);       // 방 근처 아님 → 고치기 불가
        }

        // Branch 2: 긴급회피 — 항상 선택 가능 (마스킹 없음)
    }

    // ===================================================================
    // Actions
    // ===================================================================

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (gameManager == null || gameManager.IsGameOver()) return;

        int destination = actions.DiscreteActions[0];    // 0~9
        int interaction = actions.DiscreteActions[1];    // 0~4
        int evade = actions.DiscreteActions[2];          // 0~1

        // ===== 긴급 회피 (최우선) =====
        if (evade == 1)
        {
            npcController.Evade();
        }
        else
        {
            // ===== 목적지 선택 =====
            if (destination < allRooms.Length && allRooms[destination] != null)
            {
                npcController.MoveToRoom(allRooms[destination]);
            }
            else if (destination == 9 && gameManager.cafePosition != null)
            {
                npcController.MoveToPosition(gameManager.cafePosition.position);
            }
            // destination == 8: 현재 위치 유지
        }

        // ===== 상호작용 =====
        switch (interaction)
        {
            case 1: // 부수기
                if (isSaboteur) npcController.TryStartSabotage();
                break;
            case 2: // 고치기
                npcController.TryStartRepair();
                break;
            case 3: // 중단
                npcController.CancelCurrentInteraction();
                break;
            case 4: // 사격
                if (isCaptain) TryShoot();
                break;
        }

        // ===== 보상 =====
        CalculateStepReward();
    }

    // ===================================================================
    // 보상 함수 (Eureka 교체 대상 영역)
    // ===================================================================

    /// <summary>
    /// 매 스텝 보상. Eureka 루프에서 이 메서드의 내용을 자동 교체합니다.
    /// </summary>
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

    /// <summary>
    /// 사보타주 완료 보상 (NPCController.CompleteInteraction에서 호출)
    /// </summary>
    public void OnSabotageComplete()
    {
        if (!isSaboteur) return;

        // 은밀 보너스: 감시자 없을 때 부수기 성공 시 추가 보상
        int watchers = CountWatchers();
        float reward = sabotageReward + (watchers == 0 ? stealthBonus : 0f);
        AddReward(reward);
    }

    /// <summary>
    /// 수리 완료 보상 (NPCController.CompleteInteraction에서 호출)
    /// </summary>
    public void OnRepairComplete()
    {
        if (isSaboteur) return;
        AddReward(repairReward);
    }

    public void OnGameEnd(bool humanWin)
    {
        if (isSaboteur)
            AddReward(humanWin ? loseReward : winReward);
        else
            AddReward(humanWin ? winReward : loseReward);
        // EndEpisode는 GameManager.NotifyGameEnd에서 일괄 호출
    }

    public void OnDeath()
    {
        AddReward(-1f);
    }

    // ===================================================================
    // 시야 유틸리티
    // ===================================================================

    bool CanSeeTarget(GameObject target)
    {
        if (target == null || target == gameObject || !target.activeInHierarchy)
            return false;

        Vector3 myPos = transform.position + Vector3.up * 1.5f;
        Vector3 targetPos = target.transform.position + Vector3.up * 1.5f;
        Vector3 dir = targetPos - myPos;

        if (dir.magnitude > visionRange) return false;

        if (Physics.Raycast(myPos, dir.normalized, out RaycastHit hit, dir.magnitude, obstacleLayer))
            return false;

        return true;
    }

    /// <summary>
    /// 나를 볼 수 있는 인간팀 캐릭터 수 (stealth 보너스 판정용)
    /// </summary>
    int CountWatchers()
    {
        int count = 0;
        if (gameManager == null || roleManager == null) return 0;

        foreach (var character in gameManager.allCharacters)
        {
            if (character == gameObject || character == null || !character.activeInHierarchy)
                continue;
            // 인간팀만 감시자로 간주
            if (roleManager.IsSaboteur(character)) continue;
            if (CanSeeTarget(character))
                count++;
        }
        return count;
    }

    // ===================================================================
    // 함장 사격
    // ===================================================================

    void TryShoot()
    {
        var captainGun = GetComponent<CaptainGun>();
        if (captainGun == null || captainGun.GetRemainingBullets() <= 0) return;

        // 시야 내 가장 가까운 캐릭터를 타겟으로 선택
        GameObject bestTarget = null;
        float bestDist = float.MaxValue;

        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var character in gameManager.allCharacters)
            {
                if (character == gameObject || character == null || !character.activeInHierarchy)
                    continue;
                if (!CanSeeTarget(character)) continue;

                float dist = Vector3.Distance(transform.position, character.transform.position);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestTarget = character;
                }
            }
        }

        if (bestTarget == null) return;

        // CaptainGun을 통해 사격 (총알 소모 + 이벤트 + 승리 체크 포함)
        captainGun.TryExecuteTarget(bestTarget);
    }

    // ===================================================================
    // Heuristic (수동 테스트용)
    // ===================================================================

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var d = actionsOut.DiscreteActions;
        d[0] = 8;   // 유지
        d[1] = 0;   // 없음
        d[2] = 0;   // 정상

        // 숫자키 1~8: 방 선택 (destination 0~7)
        for (int i = 0; i < 8; i++)
        {
            if (Input.GetKey(KeyCode.Alpha1 + i))
            {
                d[0] = i;
                break;
            }
        }

        // 9키: 카페 (destination 9), 0키: 유지 (destination 8)
        if (Input.GetKey(KeyCode.Alpha9)) d[0] = 9;   // 카페
        if (Input.GetKey(KeyCode.Alpha0)) d[0] = 8;   // 유지

        if (Input.GetKey(KeyCode.E))     d[1] = 2;   // 고치기
        if (Input.GetKey(KeyCode.Q))     d[1] = 1;   // 부수기
        if (Input.GetKey(KeyCode.X))     d[1] = 3;   // 중단
        if (Input.GetKey(KeyCode.F))     d[1] = 4;   // 사격
        if (Input.GetKey(KeyCode.Space)) d[2] = 1;   // 긴급 이탈
    }
}
