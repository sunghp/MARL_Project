using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
public class NPCController : MonoBehaviour
{
    // ===== 컴포넌트 참조 =====
    private NavMeshAgent agent;
    private NPCAIBrain aiBrain;

    // ===== 이동 설정 =====
    [Header("=== 이동 설정 ===")]
    public List<Transform> patrolPoints = new List<Transform>();
    public List<Transform> interactionPoints = new List<Transform>();
    
    private int currentPatrolIndex = 0;
    private Transform currentTarget;

    // ===== 상태 =====
    public enum NPCState
    {
        Idle,           // 대기
        Moving,         // 이동 중
        Interacting,    // 상호작용 중
        Patrolling,     // 순찰 중 (분산 순찰)
        GoingToCafe     // 카페로 이동 중 (소집)
    }

    [Header("=== 상태 (읽기 전용) ===")]
    [SerializeField] private NPCState currentState = NPCState.Idle;
    [SerializeField] private bool isDead = false;

    // ===== AI 설정 =====
    [Header("=== AI 설정 ===")]
    [Tooltip("점수 기반 AI 사용 여부")]
    public bool useScoreBasedAI = true;

    // ===== 상호작용 =====
    [Header("=== 상호작용 설정 ===")]
    public float interactionRange = 2f;
    
    private InteractionPoint currentInteractionPoint;
    private float interactionTimer = 0f;
    private bool isInteracting = false;
    private bool seenDuringSabotage = false;   // 부수는 동안 한 번이라도 목격됐는지 (구멍3: 완료 스냅샷 대신 전 구간)

    // ===== AI 행동 타이머 =====
    [Header("=== AI 타이머 ===")]
    public float minIdleTime = 1f;
    public float maxIdleTime = 3f;
    public float decisionInterval = 2f;

    private float idleTimer = 0f;

    [Header("=== 사보타주 진행 여부 ===")]
    public bool isSabotaging = false;

    // ===== ML-Agents 모드 =====
    private bool useMLAgents = false;

    // ===== ML-Agents 연동용 =====

    /// <summary>
    /// 임의 좌표로 이동 (카페 등 InteractionPoint가 아닌 위치)
    /// </summary>
    public void MoveToPosition(Vector3 position)
    {
        if (agent == null) return;
        currentInteractionPoint = null;
        agent.isStopped = false;
        agent.SetDestination(position);
        SetState(NPCState.Moving);
    }

    /// <summary>
    /// 진행 중인 상호작용 즉시 취소
    /// </summary>
    public void CancelCurrentInteraction()
    {
        if (!isInteracting) return;

        if (currentInteractionPoint != null)
        {
            currentInteractionPoint.CancelInteraction(gameObject);
        }

        isInteracting = false;
        interactionTimer = 0f;
        isSabotaging = false;
        currentInteractionPoint = null;
        agent.isStopped = false;
        SetState(NPCState.Idle);

        Debug.Log($"[ML] {gameObject.name} 상호작용 중단");
    }

    /// <summary>
    /// 긴급 회피: 가장 가까운 캐릭터 반대 방향으로 이동
    /// </summary>
    public void Evade()
    {
        if (agent == null || GameManager.Instance == null) return;

        // 진행 중인 상호작용 취소
        if (isInteracting)
            CancelCurrentInteraction();

        // 가장 가까운 다른 캐릭터 찾기
        GameObject nearest = null;
        float nearestDist = float.MaxValue;

        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character == gameObject || character == null || !character.activeInHierarchy)
                continue;

            float dist = Vector3.Distance(transform.position, character.transform.position);
            if (dist < nearestDist)
            {
                nearestDist = dist;
                nearest = character;
            }
        }

        if (nearest != null)
        {
            // 반대 방향으로 일정 거리 이동
            Vector3 awayDir = (transform.position - nearest.transform.position).normalized;
            Vector3 evadeTarget = transform.position + awayDir * 8f;

            agent.isStopped = false;
            agent.SetDestination(evadeTarget);
            SetState(NPCState.Moving);
            currentInteractionPoint = null;
        }
    }

    public void MoveToRoom(InteractionPoint room)
    {
        if (room == null || agent == null) return;

        // 진행 중인 상호작용이 있으면 제대로 취소 (방 점유 해제)
        if (isInteracting)
            CancelCurrentInteraction();

        currentInteractionPoint = room;
        agent.isStopped = false;
        agent.SetDestination(room.transform.position);
        SetState(NPCState.Moving);
    }

    public bool IsNearInteractionPoint()
    {
        if (currentInteractionPoint == null) return false;

        Vector3 a = transform.position;
        Vector3 b = currentInteractionPoint.transform.position;
        a.y = 0f;
        b.y = 0f;

        return Vector3.Distance(a, b) <= interactionRange;
    }

    public void TryStartSabotage()
    {
        if (currentInteractionPoint != null && IsNearInteractionPoint())
        {
            if (!currentInteractionPoint.TryStartInteraction(gameObject)) return;
            isSabotaging = true;
            StartInteraction();
        }
    }

    public void TryStartRepair()
    {
        bool near = IsNearInteractionPoint();
        if (currentInteractionPoint != null && near)
        {
            if (!currentInteractionPoint.TryStartInteraction(gameObject, true))
            {
                return;
            }
            isSabotaging = false;
            StartInteraction();
        }
    }

    // ===== Unity 생명주기 =====
    void Start()
    {
        // NavMeshAgent 가져오기
        agent = GetComponent<NavMeshAgent>();
        if (agent == null)
        {
            agent = gameObject.AddComponent<NavMeshAgent>();
        }

        // ML-Agents 에이전트가 있으면 AI Brain 자동 의사결정 비활성화
        if (GetComponent<NPCAgent>() != null)
        {
            useMLAgents = true;
            useScoreBasedAI = false;
        }

        // NPCAIBrain 가져오기 (ML-Agents 미사용 시에만 추가)
        aiBrain = GetComponent<NPCAIBrain>();
        if (aiBrain == null && useScoreBasedAI)
        {
            aiBrain = gameObject.AddComponent<NPCAIBrain>();
        }

        // 속도 설정
        if (GameManager.Instance != null)
        {
            agent.speed = GameManager.Instance.moveSpeed;
        }

        // 초기 상태
        currentState = NPCState.Idle;
        idleTimer = Random.Range(minIdleTime, maxIdleTime);
    }

    // ===== 제어 모드 토글 (평가 모드용) =====
    // useML=true  → ML-Agents 학습 정책이 제어 (OnActionReceived 실행)
    // useML=false → 규칙봇(NPCAIBrain + 상태머신)이 제어
    public bool IsUsingML() => useMLAgents;

    public void SetControlMode(bool useML)
    {
        useMLAgents = useML;
        useScoreBasedAI = !useML;

        if (!useML)
        {
            // 규칙봇 모드: 브레인 확보 + 즉시 의사결정하도록 Idle 초기화
            if (aiBrain == null) aiBrain = GetComponent<NPCAIBrain>();
            if (aiBrain == null) aiBrain = gameObject.AddComponent<NPCAIBrain>();
        }

        // 상태 초기화 (Start()보다 먼저 불려도 안전하게 null 가드)
        currentState = NPCState.Idle;
        idleTimer = 0f;
    }

    void Update()
    {
        if (isDead) return;

        // 게임 상태 체크
        if (GameManager.Instance != null)
        {
            if (GameManager.Instance.currentState == GameManager.GameState.Meeting)
            {
                if (currentState != NPCState.GoingToCafe)
                {
                    agent.isStopped = true;
                }
                return;
            }
            else if (GameManager.Instance.currentState != GameManager.GameState.Playing)
            {
                agent.isStopped = true;
                return;
            }
        }

        TryAutoRepair();

        // 상태에 따른 행동
        switch (currentState)
        {
            case NPCState.Idle:
                HandleIdle();
                break;
            case NPCState.Moving:
                HandleMoving();
                break;
            case NPCState.Patrolling:
                HandlePatrolling();
                break;
            case NPCState.Interacting:
                HandleInteracting();
                break;
            case NPCState.GoingToCafe:
                HandleGoingToCafe();
                break;
        }
    }

    /// <summary>
    /// 이동 중 재결정이 필요한지 체크 (알람 등 긴급 상황)
    /// </summary>
    bool ShouldReconsider()
    {
        // 사보타주는 재결정 안 함 (타겟 유지)
        if (RoleManager.Instance != null && RoleManager.Instance.IsSaboteur(gameObject))
        {
            return false;
        }
        
        NPCAIBrain brain = GetComponent<NPCAIBrain>();
        if (brain == null) return false;
        
        HashSet<string> damagedRooms = brain.GetKnownDamagedRooms();
        
        // 알람 뜬 방이 있는데
        if (damagedRooms.Count > 0)
        {
            // 현재 목적지가 없거나
            if (currentInteractionPoint == null) return true;
            
            // 현재 목적지가 알람 방이 아니면 재결정
            if (!damagedRooms.Contains(currentInteractionPoint.roomName))
            {
                Debug.Log($"[재결정] {gameObject.name} - 알람 감지, 목적지 변경 필요");
                return true;
            }
        }
        
        return false;
    }

    // ===== 상태별 처리 =====
    void HandleIdle()
    {
        // ML-Agents 모드: 에이전트가 OnActionReceived로 직접 제어하므로 자동 의사결정 안함
        if (useMLAgents) return;

        idleTimer -= Time.deltaTime;

        if (idleTimer <= 0f)
        {
            DecideNextAction();
        }
    }

    void HandleMoving()
    {
        if (agent.pathPending) return;

        // ML-Agents 모드가 아닐 때만 자동 재결정
        if (!useMLAgents && ShouldReconsider())
        {
            currentInteractionPoint = null;
            DecideNextAction();
            return;
        }

        // 목적지 도착 체크
        if (agent.remainingDistance <= agent.stoppingDistance + 0.5f)
        {
            if (useMLAgents)
            {
                // ML-Agents 모드: 도착만 알림, 상호작용은 에이전트가 별도 행동으로 결정
                SetState(NPCState.Idle);
            }
            else if (currentInteractionPoint != null)
            {
                // 기존 AI: 상호작용 포인트에 도착했으면 상호작용 시작
                TryStartInteraction();
            }
            else
            {
                // 도착했는데 상호작용 포인트가 없으면 대기
                SetState(NPCState.Idle);
                idleTimer = Random.Range(minIdleTime, maxIdleTime);
            }
        }
    }

    void HandlePatrolling()
    {
        if (agent.pathPending) return;

        // ML-Agents 모드가 아닐 때만 자동 재결정
        if (!useMLAgents && ShouldReconsider())
        {
            DecideNextAction();
            return;
        }

        // 순찰 목적지 도착
        if (agent.remainingDistance <= agent.stoppingDistance + 0.5f)
        {
            SetState(NPCState.Idle);
            idleTimer = Random.Range(minIdleTime, maxIdleTime);
        }
    }

    void HandleInteracting()
    {
        interactionTimer += Time.deltaTime;

        // 부수는 동안 한 번이라도 목격되면 은닉 실패로 기록 (구멍3) + 목격자별 기록 갱신
        if (isSabotaging && GameManager.Instance != null)
        {
            if (GameManager.Instance.RecordWitnesses(gameObject) > 0)
                seenDuringSabotage = true;
        }

        float requiredTime = GetInteractionTime();
        if (interactionTimer >= requiredTime)
        {
            CompleteInteraction();
        }
    }

    void HandleGoingToCafe()
    {
        if (agent.pathPending) return;

        // 카페 도착 체크
        if (agent.remainingDistance <= agent.stoppingDistance)
        {
            agent.isStopped = true;
            Debug.Log($"[소집 완료] {gameObject.name}이(가) 카페에 도착했습니다.");
        }
    }

    // ===== AI 의사결정 =====
    void DecideNextAction()
    {
        if (useScoreBasedAI && aiBrain != null)
        {
            DecideWithScoreBasedAI();
        }
        else
        {
            DecideWithRandomAI();
        }
    }

    /// <summary>
    /// 점수 기반 AI로 의사결정
    /// </summary>
    void DecideWithScoreBasedAI()
    {
        InteractionPoint targetRoom = aiBrain.DecideBestRoom();

        if (targetRoom != null)
        {
            // 방으로 이동
            MoveToInteractionPoint(targetRoom);
        }
        else
        {
            // 순찰 모드
            StartPatrol();
        }
    }

    /// <summary>
    /// 순찰 시작 (분산 순찰)
    /// </summary>
    void StartPatrol()
    {
        Vector3? patrolPos = aiBrain?.GetPatrolPosition();

        if (patrolPos.HasValue)
        {
            agent.isStopped = false;
            agent.SetDestination(patrolPos.Value);
            SetState(NPCState.Patrolling);
            currentInteractionPoint = null;
        }
        else if (patrolPoints.Count > 0)
        {
            // 분산 위치 못 찾으면 기존 순찰 포인트 사용
            MoveToNextPatrolPoint();
        }
        else
        {
            // 순찰 포인트도 없으면 대기
            idleTimer = Random.Range(minIdleTime, maxIdleTime);
        }
    }

    /// <summary>
    /// 특정 InteractionPoint로 이동
    /// </summary>
    void MoveToInteractionPoint(InteractionPoint point)
    {
        currentInteractionPoint = point;
        agent.isStopped = false;
        agent.SetDestination(point.transform.position);
        SetState(NPCState.Moving);
    }

    /// <summary>
    /// 기존 랜덤 AI (fallback)
    /// </summary>
    void DecideWithRandomAI()
    {
        bool isSaboteur = false;
        if (RoleManager.Instance != null)
        {
            isSaboteur = RoleManager.Instance.IsSaboteur(gameObject);
        }

        float actionChance = Random.Range(0f, 1f);

        if (actionChance < 0.7f && interactionPoints.Count > 0)
        {
            MoveToRandomInteractionPoint();
        }
        else if (patrolPoints.Count > 0)
        {
            MoveToNextPatrolPoint();
        }
        else
        {
            idleTimer = Random.Range(minIdleTime, maxIdleTime);
        }
    }

    void MoveToRandomInteractionPoint()
    {
        if (interactionPoints.Count == 0) return;

        int randomIndex = Random.Range(0, interactionPoints.Count);
        Transform target = interactionPoints[randomIndex];

        currentInteractionPoint = target.GetComponent<InteractionPoint>();

        MoveTo(target.position);
    }

    void MoveToNextPatrolPoint()
    {
        if (patrolPoints.Count == 0) return;

        currentPatrolIndex = (currentPatrolIndex + 1) % patrolPoints.Count;
        Transform target = patrolPoints[currentPatrolIndex];

        currentInteractionPoint = null;
        agent.isStopped = false;
        agent.SetDestination(target.position);
        SetState(NPCState.Patrolling);
    }

    void MoveTo(Vector3 destination)
    {
        agent.isStopped = false;
        agent.SetDestination(destination);
        SetState(NPCState.Moving);
    }

    // ===== 상호작용 =====
    void TryStartInteraction()
    {
        if (currentInteractionPoint == null)
        {
            SetState(NPCState.Idle);
            idleTimer = Random.Range(minIdleTime, maxIdleTime);
            return;
        }

        // 상호작용 시작 시도 (규칙봇: 사보타주가 아니면 수리)
        bool wantsRepair = RoleManager.Instance == null || !RoleManager.Instance.IsSaboteur(gameObject);
        if (currentInteractionPoint.TryStartInteraction(gameObject, wantsRepair))
        {
            // 규칙봇 자동 경로: 역할 기반으로 부수기/고치기 결정
            // (사보타주는 위험 시 순찰하므로, 이 경로엔 부수기 타겟에 도착했을 때만 옴)
            isSabotaging = RoleManager.Instance != null && RoleManager.Instance.IsSaboteur(gameObject);
            StartInteraction();
        }
        else
        {
            // 실패 시 다른 행동 결정
            Debug.Log($"[상호작용 실패] {gameObject.name} - {currentInteractionPoint.roomName} 사용 중, 다른 행동 결정");
            currentInteractionPoint = null;
            SetState(NPCState.Idle);
            idleTimer = 0.5f;  // 짧은 대기 후 재결정
        }
    }

    void StartInteraction()
    {
        isInteracting = true;
        interactionTimer = 0f;
        agent.isStopped = true;
        SetState(NPCState.Interacting);

        // 부수기 시작 시점의 목격 여부로 초기화 (이후 매 프레임 갱신)
        if (isSabotaging && GameManager.Instance != null)
            seenDuringSabotage = GameManager.Instance.RecordWitnesses(gameObject) > 0;
        else
            seenDuringSabotage = false;

#if UNITY_EDITOR
        Debug.Log($"[NPC 상호작용 시작] {gameObject.name} → {currentInteractionPoint.roomName}");
#endif
    }

    void CompleteInteraction()
    {
        if (currentInteractionPoint != null)
        {
            // isSabotaging 플래그 사용: 에이전트의 행동 선택이 실제 결과에 반영됨
            float healthChange = currentInteractionPoint.OnInteractionComplete(gameObject, isSabotaging);

            // 보상/지표: 부수는 동안 전 구간 은닉 여부(seenDuringSabotage) 사용
            NPCAgent npcAgent = GetComponent<NPCAgent>();
            if (isSabotaging)
            {
                bool wasHidden = !seenDuringSabotage;
                if (GameManager.Instance != null) GameManager.Instance.RecordSabotage(wasHidden);
                if (npcAgent != null) npcAgent.OnSabotageComplete(wasHidden);
            }
            else
            {
                // 실제로 회복된 경우만 수리로 인정 (만땅인 방 수리 파밍 방지)
                if (healthChange > 0f)
                {
                    float repairAmount = GameManager.Instance != null ? GameManager.Instance.repairAmount : 15f;
                    if (GameManager.Instance != null) GameManager.Instance.RecordRepair();
                    if (npcAgent != null) npcAgent.OnRepairComplete(healthChange / repairAmount);
                }
            }
        }

        isInteracting = false;
        interactionTimer = 0f;
        isSabotaging = false;
        currentInteractionPoint = null;

        SetState(NPCState.Idle);
        idleTimer = Random.Range(minIdleTime, maxIdleTime);

#if UNITY_EDITOR
        Debug.Log($"[NPC 상호작용 완료] {gameObject.name}");
#endif
    }

    float GetInteractionTime()
    {
        if (GameManager.Instance == null) return 3f;

        // 역할이 아닌 실제 행동 선택(isSabotaging)에 따라 소요 시간 결정
        return isSabotaging ? GameManager.Instance.sabotageTime : GameManager.Instance.repairTime;
    }

    // ===== 카페 소집 =====
    public void TeleportToCafe()
    {
        if (GameManager.Instance.cafePosition == null) return;

        // 진행 중인 상호작용 취소
        if (isInteracting && currentInteractionPoint != null)
        {
            currentInteractionPoint.CancelInteraction(gameObject);
            isInteracting = false;
            interactionTimer = 0f;
            currentInteractionPoint = null;
        }

        // 카페로 이동
        agent.isStopped = false;
        agent.SetDestination(GameManager.Instance.cafePosition.position);
        SetState(NPCState.GoingToCafe);

        Debug.Log($"[소집] {gameObject.name}이(가) 카페로 이동합니다.");
    }

    public void TeleportToCafeInstant()
    {
        if (GameManager.Instance.cafePosition == null) return;

        agent.enabled = false;
        transform.position = GameManager.Instance.cafePosition.position;
        agent.enabled = true;

        if (isInteracting && currentInteractionPoint != null)
        {
            currentInteractionPoint.CancelInteraction(gameObject);
            isInteracting = false;
            interactionTimer = 0f;
            currentInteractionPoint = null;
        }

        SetState(NPCState.Idle);
        Debug.Log($"[즉시 소집] {gameObject.name}이(가) 카페로 텔레포트했습니다.");
    }

    // ===== 사망 처리 =====
    public void Die()
    {
        isDead = true;
        agent.isStopped = true;

        // 진행 중인 상호작용 취소
        if (isInteracting && currentInteractionPoint != null)
        {
            currentInteractionPoint.CancelInteraction(gameObject);
        }

        // ML-Agents 사망 패널티
        NPCAgent npcAgent = GetComponent<NPCAgent>();
        if (npcAgent != null)
        {
            npcAgent.OnDeath();
        }

        Debug.Log($"[NPC 사망] {gameObject.name}");

        GameManager.Instance.RemoveCharacter(gameObject);

        gameObject.SetActive(false);
    }

    // ===== 상태 변경 =====
    void SetState(NPCState newState)
    {
        currentState = newState;
    }

    // ===== Getter =====
    public bool IsDead() => isDead;
    public bool IsInteracting() => isInteracting;
    public NPCState GetCurrentState() => currentState;
    public bool IsMovingToRoom() => currentState == NPCState.Moving && currentInteractionPoint != null;

    // ===== 자동 수리 (GameManager.autoRepair) =====
    // 인간팀 ML 에이전트가 손상된 방 근처(interactionRange)에 있으면 행동 선택 없이 수리 시작.
    // 사보타주가 부수는 중인 방이면 InteractionPoint가 부수기를 끊는다.
    private InteractionPoint[] cachedRooms;

    void TryAutoRepair()
    {
        if (!useMLAgents || isInteracting || isDead) return;
        if (GameManager.Instance == null || !GameManager.Instance.autoRepair) return;
        if (RoleManager.Instance == null || RoleManager.Instance.IsSaboteur(gameObject)) return;

        if (cachedRooms == null) cachedRooms = FindObjectsOfType<InteractionPoint>();
        InteractionPoint best = null;
        float bestDist = float.MaxValue;
        foreach (var room in cachedRooms)
        {
            if (room == null || room.GetCurrentHealth() >= room.maxHealth) continue;
            Vector3 a = transform.position, b = room.transform.position;
            a.y = 0f; b.y = 0f;
            float d = Vector3.Distance(a, b);
            if (d > interactionRange || d >= bestDist) continue;
            GameObject user = room.GetCurrentUser();
            bool usable = !room.IsBeingUsed() ||
                          (user != null && user.GetComponent<NPCController>() != null &&
                           user.GetComponent<NPCController>().isSabotaging);
            if (!usable) continue;
            best = room;
            bestDist = d;
        }
        if (best == null) return;

        currentInteractionPoint = best;
        TryStartRepair();
    }

    /// <summary>
    /// 상호작용 진행률 반환 (0~1). 상호작용 중이 아니면 0.
    /// </summary>
    public float GetInteractionProgress()
    {
        if (!isInteracting) return 0f;
        float requiredTime = GetInteractionTime();
        return requiredTime > 0f ? Mathf.Clamp01(interactionTimer / requiredTime) : 0f;
    }

    // ===== 외부에서 포인트 설정 =====
    public void SetInteractionPoints(List<Transform> points)
    {
        interactionPoints = points;
    }

    public void SetPatrolPoints(List<Transform> points)
    {
        patrolPoints = points;
    }

    // ===== NPC 리셋 =====

    public void ResetNPC()
    {
        // 상태 초기화
        isDead = false;
        isSabotaging = false;
        isInteracting = false;
        currentState = NPCState.Idle;
        idleTimer = 0f;
        interactionTimer = 0f;
        currentInteractionPoint = null;

        // NavMeshAgent 정지
        if (agent != null)
        {
            agent.isStopped = false;
            agent.ResetPath();
            agent.velocity = Vector3.zero;
        }

#if UNITY_EDITOR
        Debug.Log($"[리셋] {gameObject.name} NPC 상태 초기화");
#endif
    }
}