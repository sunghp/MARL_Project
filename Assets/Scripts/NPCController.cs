using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using Unity.VisualScripting;

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

    // ===== AI 행동 타이머 =====
    [Header("=== AI 타이머 ===")]
    public float minIdleTime = 1f;
    public float maxIdleTime = 3f;
    public float decisionInterval = 2f;

    private float idleTimer = 0f;

    [Header("=== 사보타주 진행 여부 ===")]
    public bool isSabotaging = false;

    // ===== ML-Agents 연동용 =====

    public void MoveToRoom(InteractionPoint room)
    {
        if (room == null || agent == null) return;
        
        currentInteractionPoint = room;
        agent.SetDestination(room.transform.position);
        SetState(NPCState.Moving);
    }

    public bool IsNearInteractionPoint()
    {
        if (currentInteractionPoint == null) return false;
        
        float distance = Vector3.Distance(transform.position, currentInteractionPoint.transform.position);
        return distance <= interactionRange;
    }

    public void TryStartSabotage()
    {
        if (currentInteractionPoint != null && IsNearInteractionPoint())
        {
            // 부수기 상호작용 시작
            isSabotaging = true;
            StartInteraction();
        }
    }

    public void TryStartRepair()
    {
        if (currentInteractionPoint != null && IsNearInteractionPoint())
        {
            // 고치기 상호작용 시작
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

        // NPCAIBrain 가져오기 (없으면 추가)
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
        idleTimer -= Time.deltaTime;

        if (idleTimer <= 0f)
        {
            DecideNextAction();
        }
    }

    void HandleMoving()
    {
        if (agent.pathPending) return;

        // 추가: 이동 중 긴급 상황 체크
        if (ShouldReconsider())
        {
            currentInteractionPoint = null;
            DecideNextAction();
            return;
        }

        // 목적지 도착 체크
        if (agent.remainingDistance <= agent.stoppingDistance + 0.5f)
        {
            // 상호작용 포인트에 도착했으면 상호작용 시작
            if (currentInteractionPoint != null)
            {
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

        // 추가: 순찰 중 긴급 상황 체크
        if (ShouldReconsider())
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

        // 상호작용 시작 시도
        if (currentInteractionPoint.TryStartInteraction(gameObject))
        {
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

        Debug.Log($"[NPC 상호작용 시작] {gameObject.name} → {currentInteractionPoint.roomName}");
    }

    void CompleteInteraction()
    {
        if (currentInteractionPoint != null)
        {
            bool isSaboteur = RoleManager.Instance.IsSaboteur(gameObject);
            currentInteractionPoint.OnInteractionComplete(gameObject, isSaboteur);
        }

        isInteracting = false;
        interactionTimer = 0f;
        currentInteractionPoint = null;

        SetState(NPCState.Idle);
        idleTimer = Random.Range(minIdleTime, maxIdleTime);

        Debug.Log($"[NPC 상호작용 완료] {gameObject.name}");
    }

    float GetInteractionTime()
    {
        if (GameManager.Instance == null) return 3f;

        bool isSaboteur = RoleManager.Instance.IsSaboteur(gameObject);
        return isSaboteur ? GameManager.Instance.sabotageTime : GameManager.Instance.repairTime;
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
        currentState = NPCState.Idle;
        idleTimer = 0f;
        interactionTimer = 0f;
        currentInteractionPoint = null;
        
        // NavMeshAgent 정지
        if (agent != null)
        {
            agent.ResetPath();
            agent.velocity = Vector3.zero;
        }
        
        Debug.Log($"[리셋] {gameObject.name} NPC 상태 초기화");
    }
}