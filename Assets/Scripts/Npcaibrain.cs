using UnityEngine;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// NPC AI 의사결정 시스템 (재설계)
/// - 인간: 분산 순찰 + 알람 시 수리
/// - 함장: 인간 + 의심 점수 관리 + 처형
/// - 사보타주: 타겟 고정 + 들키지 않게 파괴
/// 
/// 핵심: NPC는 전지적 시점이 아님 - 알람으로만 방 상태를 앎
/// </summary>
public class NPCAIBrain : MonoBehaviour
{
    // ===== 시야 설정 =====
    [Header("=== 시야 설정 ===")]
    [Tooltip("시야 거리")]
    public float visionRange = 15f;
    
    [Tooltip("시야 체크용 레이어 마스크 (장애물)")]
    public LayerMask obstacleLayer;

    // ===== 분산 순찰 설정 =====
    [Header("=== 순찰 설정 ===")]
    [Tooltip("다른 NPC와 유지하려는 최소 거리")]
    public float preferredNPCDistance = 15f;

    // ===== 사보타주 설정 =====
    [Header("=== 사보타주 설정 ===")]
    [Tooltip("목격 위험 시 회피할 확률 (0~1)")]
    public float avoidanceChance = 0.8f;
    
    [Tooltip("위험 판단 기준 (시야 내 캐릭터 수)")]
    public int dangerThreshold = 1;

    // ===== 함장 전용 설정 =====
    [Header("=== 함장 전용 설정 ===")]
    [Tooltip("의심 점수 임계값")]
    public float suspicionThreshold = 100f;
    
    [Tooltip("알람 근처 목격 시 의심 증가량")]
    public float alarmWitnessSuspicion = 30f;
    
    [Tooltip("부수기 직접 목격 시 의심 증가량")]
    public float directSabotageSuspicion = 50f;
    
    [Tooltip("부수기 현장 근처 목격 시 의심 증가량")]
    public float nearbySabotageSuspicion = 20f;
    
    [Tooltip("반복 목격 시 추가 의심")]
    public float repeatWitnessSuspicion = 15f;
    
    [Tooltip("처형 결정까지 대기 시간")]
    public float executionDecisionDelay = 3f;

    // ===== 알람 기억 시스템 =====
    private HashSet<string> knownDamagedRooms = new HashSet<string>();  // 알람으로 알게 된 손상된 방
    
    // ===== 사보타주 타겟 시스템 =====
    private InteractionPoint sabotageTarget = null;  // 현재 공략 중인 방

    // ===== 의심 점수 (함장용) =====
    private Dictionary<GameObject, float> suspicionScores = new Dictionary<GameObject, float>();
    private Dictionary<GameObject, int> alarmWitnessCount = new Dictionary<GameObject, int>();

    // ===== 함장 처형 =====
    private float executionCheckTimer = 0f;
    private bool isConsideringExecution = false;
    private CaptainGun captainGun;

    // ===== 캐싱 =====
    private InteractionPoint[] allRooms;
    private NPCController npcController;

    // ===== 현재 목적지 (순찰/이동 포함) =====

    private InteractionPoint currentDestination = null;

    // ===== Unity 생명주기 =====
    void Start()
    {
        npcController = GetComponent<NPCController>();
        allRooms = FindObjectsOfType<InteractionPoint>();
        
        if (GameManager.Instance != null)
        {
            visionRange = GameManager.Instance.visionRange;
        }
        
        // 이벤트 구독
        if (SystemHealth.Instance != null)
        {
            SystemHealth.Instance.OnLocationAlert += OnRoomAlert;
        }
        InteractionPoint.OnSabotageDetected += OnSabotageDetected;
        InteractionPoint.OnRoomRepaired += OnRoomRepaired;
        
        // 함장용 초기화
        captainGun = GetComponent<CaptainGun>();
        InitializeSuspicionScores();
    }

    void Update()
    {
        // ML-Agents 모드: 에이전트가 직접 제어하므로 AI 브레인 비활성화
        if (GetComponent<NPCAgent>() != null) return;

        // 함장 CaptainGun 참조 갱신 (타이밍 문제 해결)
        if (captainGun == null && RoleManager.Instance != null && RoleManager.Instance.IsCaptain(gameObject))
        {
            captainGun = GetComponent<CaptainGun>();
        }

        // 함장만 처형 로직 실행
        if (RoleManager.Instance != null && RoleManager.Instance.IsCaptain(gameObject))
        {
            UpdateCaptainExecution();
        }
    }

    void OnDestroy()
    {
        if (SystemHealth.Instance != null)
        {
            SystemHealth.Instance.OnLocationAlert -= OnRoomAlert;
        }
        InteractionPoint.OnSabotageDetected -= OnSabotageDetected;
        InteractionPoint.OnRoomRepaired -= OnRoomRepaired;
    }

    // ========================================
    // 알람 기억 시스템
    // ========================================
    
    void OnRoomAlert(string roomName)
    {
        // 모든 NPC가 알람을 들음
        knownDamagedRooms.Add(roomName);
        Debug.Log($"🚨 [{gameObject.name}] 알람 수신: {roomName} 손상됨!");
        
        // 함장은 추가로 의심 점수 계산
        if (RoleManager.Instance != null && RoleManager.Instance.IsCaptain(gameObject))
        {
            ProcessSuspicionOnAlert(roomName);
        }
    }

    void OnRoomRepaired(string roomName)
    {
        // 수리 완료 알림 - 기억에서 제거
        if (knownDamagedRooms.Contains(roomName))
        {
            knownDamagedRooms.Remove(roomName);
            Debug.Log($"✓ [{gameObject.name}] {roomName} 수리 완료 인지");
        }
    }

    // ========================================
    // 메인 의사결정
    // ========================================
    
    /// <summary>
    /// 역할에 따라 다음 행동 결정
    /// 반환값: 이동할 방 (null이면 순찰)
    /// </summary>
    public InteractionPoint DecideBestRoom()
    {
        if (allRooms == null || allRooms.Length == 0) return null;

        RoleManager.Role myRole = RoleManager.Instance.GetRole(gameObject);

        switch (myRole)
        {
            case RoleManager.Role.Saboteur:
                return DecideAsSaboteur();
            case RoleManager.Role.Captain:
                return DecideAsCaptain();
            case RoleManager.Role.Human:
            default:
                return DecideAsHuman();
        }
    }

    /// <summary>
    /// 순찰할 위치 반환 (분산 순찰)
    /// </summary>
    public Vector3? GetPatrolPosition()
    {
        Dictionary<InteractionPoint, float> roomScores = new Dictionary<InteractionPoint, float>();
        
        foreach (var room in allRooms)
        {
            if (room.IsBeingUsed()) continue;
            
            float dispersionScore = CalculateDispersionScore(room.transform.position);
            
            // 다른 NPC 근처/이동중이면 점수 감소 (패널티 강화)
            int othersNearby = CountNPCsNearPosition(room.transform.position, 10f);
            int othersHeading = CountNPCsHeadingToRoom(room);
            
            // 패널티 강화: 1명당 50% 감소
            float crowdPenalty = 1f / Mathf.Pow(1.5f, othersNearby + othersHeading);
            
            float finalScore = dispersionScore * crowdPenalty;
            
            if (finalScore > 0)
            {
                roomScores[room] = finalScore;
            }
        }
        
        InteractionPoint selected = SelectRoomByProbability(roomScores, 1.5f);
        
        if (selected != null)
        {
            // 목적지 저장 (다른 NPC가 참조할 수 있도록)
            SetCurrentDestination(selected);
            
            Debug.Log($"[순찰] {gameObject.name} → {selected.roomName} (확률 선택, 분산 점수: {roomScores[selected]:F1})");
            return selected.transform.position;
        }
        
        currentDestination = null;
        return null;
    }

    // ========================================
    // 인간 AI
    // ========================================
    
    InteractionPoint DecideAsHuman()
    {
        // 1. 알람 뜬 방들 점수 계산
        if (knownDamagedRooms.Count > 0)
        {
            Dictionary<InteractionPoint, float> roomScores = new Dictionary<InteractionPoint, float>();
            
            foreach (string roomName in knownDamagedRooms)
            {
                InteractionPoint room = FindRoomByName(roomName);
                if (room == null) continue;
                if (room.IsBeingUsed()) continue;
                
                float distance = Vector3.Distance(transform.position, room.transform.position);
                float distanceScore = 100f / Mathf.Max(distance, 1f);
                
                // 다른 NPC가 이미 가고 있으면 점수 감소 (패널티 강화)
                int othersHeading = CountNPCsHeadingToRoom(room);
                int othersNearby = CountNPCsNearPosition(room.transform.position);
                float crowdPenalty = 1f / Mathf.Pow(1.5f, othersHeading + othersNearby);
                
                float finalScore = distanceScore * crowdPenalty;
                roomScores[room] = finalScore;
                
                Debug.Log($"[인간 점수] {gameObject.name} → {room.roomName}: 거리={distanceScore:F1}, 혼잡패널티={crowdPenalty:F2}, 최종={finalScore:F1}");
            }
            
            if (roomScores.Count > 0)
            {
                InteractionPoint selected = SelectRoomByProbability(roomScores, 0.5f);
                if (selected != null)
                {
                    // 목적지 저장
                    SetCurrentDestination(selected);
                    
                    Debug.Log($"[인간 AI] {gameObject.name} → {selected.roomName} (확률 선택, 알람 대응)");
                    return selected;
                }
            }
        }
        
        // 2. 알람 없으면 순찰
        currentDestination = null;
        Debug.Log($"[인간 AI] {gameObject.name} → 순찰 모드");
        return null;
    }

    InteractionPoint GetClosestAlertRoom()
    {
        InteractionPoint closest = null;
        float closestDist = float.MaxValue;

        foreach (string roomName in knownDamagedRooms)
        {
            InteractionPoint room = FindRoomByName(roomName);
            if (room == null) continue;
            if (room.IsBeingUsed()) continue;  // 다른 NPC가 수리 중이면 패스
            
            float dist = Vector3.Distance(transform.position, room.transform.position);
            if (dist < closestDist)
            {
                closestDist = dist;
                closest = room;
            }
        }

        return closest;
    }

    // ========================================
    // 함장 AI
    // ========================================
    
    InteractionPoint DecideAsCaptain()
    {
        // 1. 알람 뜬 방이 있으면 가장 가까운 곳으로 (수리 + 목격자 확인)
        InteractionPoint alertRoom = GetClosestAlertRoom();
        if (alertRoom != null)
        {
            Debug.Log($"[함장 AI] {gameObject.name} → {alertRoom.roomName} (알람 대응 + 감시)");
            return alertRoom;
        }
        
        // 2. 의심 높은 캐릭터가 있으면 따라다니기 (TODO: 추후 구현)
        
        // 3. 알람 없으면 순찰
        Debug.Log($"[함장 AI] {gameObject.name} → 순찰 모드");
        return null;
    }

    // ========================================
    // 사보타주 AI
    // ========================================
    
    InteractionPoint DecideAsSaboteur()
    {
        // 1. 현재 타겟이 있고 유효하면 계속 공략
        if (sabotageTarget != null && IsTargetValid(sabotageTarget))
        {
            // 목격 위험 체크
            if (IsInDanger(sabotageTarget.transform.position))
            {
                Debug.Log($"[사보타주 AI] {gameObject.name} - 위험 감지! 회피");
                return null;  // 순찰 척
            }
            
            Debug.Log($"[사보타주 AI] {gameObject.name} → {sabotageTarget.roomName} (타겟 유지)");
            return sabotageTarget;
        }
        
        // 2. 새 타겟 선정
        sabotageTarget = SelectNewTarget();
        
        if (sabotageTarget != null)
        {
            Debug.Log($"[사보타주 AI] {gameObject.name} → {sabotageTarget.roomName} (새 타겟)");
            return sabotageTarget;
        }
        
        // 3. 적절한 타겟 없으면 순찰 척
        Debug.Log($"[사보타주 AI] {gameObject.name} → 순찰 척 (타겟 없음)");
        return null;
    }

    bool IsTargetValid(InteractionPoint target)
    {
        if (target == null) return false;
        if (target.GetHealthPercent() <= 0f) return false;  // 이미 완파
        if (target.IsBeingUsed() && target.GetCurrentUser() != gameObject) return false;  // 다른 사람이 사용 중
        return true;
    }

    InteractionPoint SelectNewTarget()
    {
        Dictionary<InteractionPoint, float> roomScores = new Dictionary<InteractionPoint, float>();
        
        foreach (var room in allRooms)
        {
            float score = CalculateSabotageScore(room);
            
            if (score > 0)
            {
                // 다른 사보타주가 이미 공략 중이면 점수 대폭 감소
                int otherSabsTargeting = 0;
                foreach (var sab in GameManager.Instance.saboteurs)
                {
                    if (sab == gameObject) continue;
                    if (sab == null || !sab.activeInHierarchy) continue;
                    
                    NPCAIBrain otherBrain = sab.GetComponent<NPCAIBrain>();
                    if (otherBrain != null && otherBrain.GetCurrentSabotageTarget() == room)
                    {
                        otherSabsTargeting++;
                    }
                }
                
                // 다른 사보타주 있으면 점수 80% 감소
                float sabPenalty = 1f / (1f + otherSabsTargeting * 4f);
                score *= sabPenalty;
                
                roomScores[room] = score;
            }
        }
        
        return SelectRoomByProbability(roomScores, 0.3f);  // 사보타주는 덜 랜덤하게 (효율 중시)
    }

    float CalculateSabotageScore(InteractionPoint room)
    {
        // 사용 중이면 제외
        if (room.IsBeingUsed()) return float.MinValue;
        
        // 이미 완파면 제외
        if (room.GetHealthPercent() <= 0f) return float.MinValue;
        
        float distance = Vector3.Distance(transform.position, room.transform.position);
        float damagePercent = 1f - room.GetHealthPercent();  // 0~1 (부서질수록 높음)
        
        // 점수 = (파괴도)^2 × (1 / 거리) × 안전도
        // 파괴도 제곱: 이미 부서진 방 강하게 선호
        // 거리 역수: 가까울수록 좋음
        // 안전도: 목격 위험 있으면 감소
        
        float damageScore = Mathf.Pow(damagePercent + 0.1f, 2);  // +0.1은 새 방도 선택되게
        float distanceScore = 1f / Mathf.Max(distance, 1f);
        float safetyScore = CalculateSafetyScore(room.transform.position);
        
        float finalScore = damageScore * distanceScore * safetyScore * 1000f;
        
        return finalScore;
    }

    float CalculateSafetyScore(Vector3 position)
    {
        int visibleCount = CountVisibleCharactersNearPosition(position);
        
        // 안전도 = 1 / (1 + 목격자수)^2
        // 목격자 0명: 1.0
        // 목격자 1명: 0.25
        // 목격자 2명: 0.11
        return 1f / Mathf.Pow(1f + visibleCount, 2);
    }

    bool IsInDanger(Vector3 position)
    {
        int visibleCount = CountVisibleCharactersNearPosition(position);
        
        if (visibleCount >= dangerThreshold)
        {
            // 회피 확률에 따라 결정
            return Random.value < avoidanceChance;
        }
        
        return false;
    }

    // ========================================
    // 확률적 방 선택 (Softmax 기반)
    // ========================================

    /// <summary>
    /// 점수 배열을 확률로 변환하여 랜덤 선택
    /// temperature가 높을수록 랜덤, 낮을수록 최고점수 선택
    /// </summary>
    InteractionPoint SelectRoomByProbability(Dictionary<InteractionPoint, float> roomScores, float temperature = 1.0f)
    {
        if (roomScores.Count == 0) return null;
        
        // 음수 점수 제거 (선택 불가 방)
        var validRooms = roomScores.Where(kvp => kvp.Value > 0).ToList();
        if (validRooms.Count == 0) return null;
        
        // 점수가 1개면 그냥 반환
        if (validRooms.Count == 1) return validRooms[0].Key;
        
        // Softmax 확률 계산
        float maxScore = validRooms.Max(kvp => kvp.Value);
        List<float> probabilities = new List<float>();
        float sumExp = 0f;
        
        foreach (var kvp in validRooms)
        {
            // 오버플로우 방지: max 빼기
            float exp = Mathf.Exp((kvp.Value - maxScore) / temperature);
            probabilities.Add(exp);
            sumExp += exp;
        }
        
        // 정규화
        for (int i = 0; i < probabilities.Count; i++)
        {
            probabilities[i] /= sumExp;
        }
        
        // 룰렛 휠 선택
        float random = Random.value;
        float cumulative = 0f;
        
        for (int i = 0; i < validRooms.Count; i++)
        {
            cumulative += probabilities[i];
            if (random <= cumulative)
            {
                return validRooms[i].Key;
            }
        }
        
        // fallback
        return validRooms[validRooms.Count - 1].Key;
    }

    /// <summary>
    /// 해당 위치 근처에 다른 NPC가 몇 명 있는지 계산
    /// </summary>
    int CountNPCsNearPosition(Vector3 position, float radius = 5f)
    {
        int count = 0;
        
        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character == gameObject) continue;
            if (character == null || !character.activeInHierarchy) continue;
            
            float dist = Vector3.Distance(position, character.transform.position);
            if (dist <= radius)
            {
                count++;
            }
        }
        
        return count;
    }

    /// <summary>
    /// 해당 방으로 이동 중인 다른 NPC가 있는지 체크
    /// </summary>
    int CountNPCsHeadingToRoom(InteractionPoint room)
    {
        int count = 0;
        
        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character == gameObject) continue;
            if (character == null || !character.activeInHierarchy) continue;
            
            NPCAIBrain otherBrain = character.GetComponent<NPCAIBrain>();
            if (otherBrain != null)
            {
                // 사보타주 타겟 체크
                if (otherBrain.GetCurrentSabotageTarget() == room)
                {
                    count++;
                    continue;
                }
                
                // 일반 목적지 체크 (순찰 포함)
                if (otherBrain.GetCurrentDestination() == room)
                {
                    count++;
                    continue;
                }
            }
            
            // 현재 상호작용 중인지 체크
            NPCController otherController = character.GetComponent<NPCController>();
            if (otherController != null && otherController.IsInteracting())
            {
                float dist = Vector3.Distance(character.transform.position, room.transform.position);
                if (dist <= 3f)
                {
                    count++;
                }
            }
        }
        
        return count;
    }

    // ========================================
    // 분산 순찰 계산
    // ========================================
    
    float CalculateDispersionScore(Vector3 position)
    {
        float totalDistance = 0f;
        int count = 0;

        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character == gameObject) continue;
            if (character == null || !character.activeInHierarchy) continue;
            
            float dist = Vector3.Distance(position, character.transform.position);
            totalDistance += dist;
            count++;
        }

        if (count == 0) return 0f;
        
        // 평균 거리가 높을수록 좋음
        return totalDistance / count;
    }

    // ========================================
    // 시야 체크
    // ========================================
    
    public bool CanSee(GameObject target)
    {
        if (target == null || target == gameObject) return false;
        if (!target.activeInHierarchy) return false;
        
        Vector3 myPos = transform.position + Vector3.up * 1.5f;
        Vector3 targetPos = target.transform.position + Vector3.up * 1.5f;
        Vector3 direction = targetPos - myPos;
        float distance = direction.magnitude;
        
        if (distance > visionRange) return false;
        
        RaycastHit hit;
        if (Physics.Raycast(myPos, direction.normalized, out hit, distance, obstacleLayer))
        {
            return false;
        }
        
        return true;
    }

    public List<GameObject> GetVisibleCharacters()
    {
        List<GameObject> visible = new List<GameObject>();
        
        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (CanSee(character))
            {
                visible.Add(character);
            }
        }
        
        return visible;
    }

    int CountVisibleCharactersNearPosition(Vector3 position)
    {
        int count = 0;
        float nearRange = 10f;
        
        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character == gameObject) continue;
            if (character == null || !character.activeInHierarchy) continue;
            
            if (!CanSee(character)) continue;
            
            float distToPosition = Vector3.Distance(character.transform.position, position);
            if (distToPosition <= nearRange)
            {
                count++;
            }
        }
        
        return count;
    }

    // ========================================
    // 함장 의심 시스템
    // ========================================
    
    void InitializeSuspicionScores()
    {
        suspicionScores.Clear();
        alarmWitnessCount.Clear();
        Invoke(nameof(DelayedInitSuspicion), 0.5f);
    }

    void DelayedInitSuspicion()
    {
        if (GameManager.Instance == null) return;
        
        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character != gameObject)
            {
                suspicionScores[character] = 0f;
                alarmWitnessCount[character] = 0;
            }
        }
    }

    void ProcessSuspicionOnAlert(string roomName)
    {
        InteractionPoint alertRoom = FindRoomByName(roomName);
        if (alertRoom == null) return;
        
        foreach (var character in GameManager.Instance.allCharacters)
        {
            if (character == gameObject) continue;
            if (character == null || !character.activeInHierarchy) continue;
            
            float distToRoom = Vector3.Distance(character.transform.position, alertRoom.transform.position);
            
            if (distToRoom <= 15f)
            {
                bool canSeeCharacter = CanSee(character);
                
                float suspicionToAdd = alarmWitnessSuspicion;
                
                if (!alarmWitnessCount.ContainsKey(character))
                {
                    alarmWitnessCount[character] = 0;
                }
                alarmWitnessCount[character]++;
                
                if (alarmWitnessCount[character] > 1)
                {
                    suspicionToAdd += repeatWitnessSuspicion * (alarmWitnessCount[character] - 1);
                }
                
                AddSuspicion(character, suspicionToAdd);
                
                string sightInfo = canSeeCharacter ? "(직접 목격)" : "(근처 추정)";
                Debug.Log($"🔍 [함장] {character.name}이(가) {roomName} 근처에서 감지됨 {sightInfo} (의심 +{suspicionToAdd:F0})");
            }
        }
    }

    void OnSabotageDetected(GameObject saboteur, string roomName, Vector3 position)
    {
        if (!RoleManager.Instance.IsCaptain(gameObject)) return;
        if (saboteur == gameObject) return;
        
        bool canSeeSaboteur = CanSee(saboteur);
        
        if (canSeeSaboteur)
        {
            AddSuspicion(saboteur, directSabotageSuspicion);
            Debug.Log($"👁️ [함장 직접 목격!] {saboteur.name}이(가) {roomName}에서 부수는 것을 봤다! (의심 +{directSabotageSuspicion})");
        }
        else
        {
            foreach (var character in GameManager.Instance.allCharacters)
            {
                if (character == gameObject) continue;
                if (character == null || !character.activeInHierarchy) continue;
                
                float distToRoom = Vector3.Distance(character.transform.position, position);
                
                if (distToRoom <= 10f && CanSee(character))
                {
                    AddSuspicion(character, nearbySabotageSuspicion);
                    Debug.Log($"🔍 [함장] {character.name}이(가) {roomName} 파괴 현장 근처에서 목격됨 (의심 +{nearbySabotageSuspicion})");
                }
            }
        }
    }

    public void AddSuspicion(GameObject target, float amount)
    {
        if (target == null || target == gameObject) return;
        
        if (!suspicionScores.ContainsKey(target))
        {
            suspicionScores[target] = 0f;
        }
        
        suspicionScores[target] += amount;
        Debug.Log($"[의심 점수] {target.name}: {suspicionScores[target]:F1}");
    }

    public GameObject GetExecutionTarget()
    {
        foreach (var kvp in suspicionScores)
        {
            if (kvp.Key == null || !kvp.Key.activeInHierarchy) continue;
            
            if (kvp.Value >= suspicionThreshold)
            {
                return kvp.Key;
            }
        }
        
        return null;
    }

    // ========================================
    // 함장 처형 로직
    // ========================================
    
    void UpdateCaptainExecution()
    {
        if (GameManager.Instance.currentState != GameManager.GameState.Playing) return;
        if (captainGun == null) return;
        if (captainGun.GetRemainingBullets() <= 0) return;
        
        GameObject target = GetExecutionTarget();
        
        if (target != null)
        {
            if (!isConsideringExecution)
            {
                isConsideringExecution = true;
                executionCheckTimer = 0f;
                Debug.Log($"⚠️ [함장 AI] {target.name} 처형 고려 중...");
            }
            else
            {
                executionCheckTimer += Time.deltaTime;
                
                if (executionCheckTimer >= executionDecisionDelay)
                {
                    PerformExecution(target);
                    isConsideringExecution = false;
                    executionCheckTimer = 0f;
                }
            }
        }
        else
        {
            isConsideringExecution = false;
            executionCheckTimer = 0f;
        }
    }

    void PerformExecution(GameObject target)
    {
        if (captainGun.GetRemainingBullets() <= 0) return;

        bool wasSaboteur = RoleManager.Instance.IsSaboteur(target);
        Debug.Log($"🔫 [함장 AI 처형] {target.name}이(가) 처형되었습니다.");

        PlayerController pc = target.GetComponent<PlayerController>();
        NPCController npc = target.GetComponent<NPCController>();

        if (pc != null) pc.Die();
        else if (npc != null) npc.Die();

        suspicionScores.Remove(target);
        alarmWitnessCount.Remove(target);

        CheckWinAfterExecution(wasSaboteur);
    }

    void CheckWinAfterExecution(bool wasSaboteur)
    {
        int humanCount = GameManager.Instance.GetAliveHumanCount();
        int saboteurCount = GameManager.Instance.GetAliveSaboteurCount();

        if (wasSaboteur)
        {
            if (saboteurCount == 0)
            {
                GameManager.Instance.HumanWin("함장이 모든 사보타주 제거");
            }
        }
        else
        {
            if (humanCount <= saboteurCount)
            {
                GameManager.Instance.SabotageWin($"함장 실수로 인간과 동수 ({humanCount} vs {saboteurCount})");
            }
        }
    }

    // ========================================
    // 유틸리티
    // ========================================
    
    InteractionPoint FindRoomByName(string roomName)
    {
        foreach (var room in allRooms)
        {
            if (room.roomName == roomName)
            {
                return room;
            }
        }
        return null;
    }

    // ========================================
    // Getter
    // ========================================
    
    public Dictionary<GameObject, float> GetAllSuspicionScores() => new Dictionary<GameObject, float>(suspicionScores);
    public HashSet<string> GetKnownDamagedRooms() => new HashSet<string>(knownDamagedRooms);
    public InteractionPoint GetCurrentSabotageTarget() => sabotageTarget;
    public List<GameObject> GetCharactersInSight() => GetVisibleCharacters();
    public void SetCurrentDestination(InteractionPoint dest)
    {
        currentDestination = dest;
    }

    public InteractionPoint GetCurrentDestination()
    {
        return currentDestination;
    }

    // ===== Brain 리셋 =====

    public void ResetBrain()
    {
        // 알람 기억 초기화
        knownDamagedRooms.Clear();

        // 의심 점수 초기화
        if (suspicionScores != null)
            suspicionScores.Clear();

        // 알람 목격 횟수 초기화
        alarmWitnessCount.Clear();

        // 타겟 초기화
        sabotageTarget = null;
        currentDestination = null;

#if UNITY_EDITOR
        Debug.Log($"[리셋] {gameObject.name} AI Brain 초기화");
#endif
    }
}
    