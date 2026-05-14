using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

public class NPCAgent : Agent
{
    [Header("참조")]
    private NPCController npcController;
    private NPCAIBrain aiBrain;
    private GameManager gameManager;
    private RoleManager roleManager;
    
    [Header("설정")]
    public float decisionInterval = 0.5f;
    
    [Header("보상 설정")]
    public float winReward = 10f;
    public float loseReward = -10f;
    public float sabotageReward = 0.5f;
    public float repairReward = 0.5f;
    public float healthChangeRewardScale = 0.01f;
    
    // 내부 상태
    private float previousAverageHealth;
    private bool isSaboteur;
    private bool isCaptain;
    private InteractionPoint[] allRooms;
    
    // ===== Observation 크기 상수 =====
    // 자기 정보: 2 (역할) + 3 (위치) = 5
    // 게임 상태: 3
    // 방 상태: 8개 방 * 3 = 24
    // 캐릭터 정보: 5명(자기 제외) * 2 = 10
    // 총합: 5 + 3 + 24 + 10 = 42
    private const int MAX_ROOMS = 8;
    private const int MAX_OTHER_CHARACTERS = 5;
    
    public override void Initialize()
    {
        npcController = GetComponent<NPCController>();
        aiBrain = GetComponent<NPCAIBrain>();
        gameManager = GameManager.Instance;
        roleManager = RoleManager.Instance;
        allRooms = FindObjectsOfType<InteractionPoint>();
        
        // 방 개수 확인 (디버그용)
        Debug.Log($"[NPCAgent] {gameObject.name} 초기화 - 감지된 방 개수: {allRooms.Length}");
    }
    
    public override void OnEpisodeBegin()
    {
        // null이면 다시 가져오기
        if (roleManager == null)
            roleManager = RoleManager.Instance;
        if (gameManager == null)
            gameManager = GameManager.Instance;
        
        // 여전히 null이면 리턴
        if (roleManager == null || gameManager == null)
        {
            Debug.LogWarning($"[NPCAgent] {gameObject.name} - Manager가 아직 초기화 안됨");
            return;
        }
        
        // 역할 확인
        isSaboteur = roleManager.IsSaboteur(gameObject);
        isCaptain = roleManager.IsCaptain(gameObject);
        
        // 초기 상태 저장
        if (SystemHealth.Instance != null)
            previousAverageHealth = SystemHealth.Instance.GetAverageHealth();
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // ===== 1. 자기 정보 (5) =====
        sensor.AddObservation(isSaboteur ? 1f : 0f);          // 1
        sensor.AddObservation(isCaptain ? 1f : 0f);           // 1
        sensor.AddObservation(transform.position.normalized); // 3 (x, y, z)
        
        // ===== 2. 게임 상태 (3) =====
        float avgHealth = 0f;
        if (SystemHealth.Instance != null)
            avgHealth = SystemHealth.Instance.GetAverageHealth();
        
        sensor.AddObservation(avgHealth / 100f);                              // 1
        sensor.AddObservation(gameManager?.GetDistanceProgress() ?? 0f);      // 1
        sensor.AddObservation(gameManager?.GetAliveHumanRatio() ?? 0f);       // 1
        
        // ===== 3. 각 방 상태 (MAX_ROOMS * 3 = 24) =====
        for (int i = 0; i < MAX_ROOMS; i++)
        {
            if (i < allRooms.Length && allRooms[i] != null)
            {
                float distance = Vector3.Distance(transform.position, allRooms[i].transform.position);
                sensor.AddObservation(distance / 50f);                    // 거리 정규화
                sensor.AddObservation(allRooms[i].GetHealthPercent() / 100f); // 방 안정도
                sensor.AddObservation(allRooms[i].IsBeingUsed() ? 1f : 0f);   // 사용 중 여부
            }
            else
            {
                // 방이 없으면 0으로 패딩
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }
        
        // ===== 4. 다른 캐릭터 정보 (MAX_OTHER_CHARACTERS * 2 = 10) =====
        // 방법: x, z 위치만 사용 (2D 평면 기준)
        int characterCount = 0;
        
        if (gameManager != null && gameManager.allCharacters != null)
        {
            foreach (var character in gameManager.allCharacters)
            {
                // 자기 자신 제외
                if (character == gameObject) continue;
                
                // 최대 5명까지만
                if (characterCount >= MAX_OTHER_CHARACTERS) break;
                
                if (character != null && character.activeInHierarchy)
                {
                    // 살아있는 캐릭터: 정규화된 위치
                    sensor.AddObservation(character.transform.position.x / 50f);
                    sensor.AddObservation(character.transform.position.z / 50f);
                }
                else
                {
                    // 죽었거나 비활성화된 캐릭터
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
                characterCount++;
            }
        }
        
        // 남은 슬롯 패딩 (캐릭터가 5명 미만일 경우)
        for (int i = characterCount; i < MAX_OTHER_CHARACTERS; i++)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
        
        // 총 Observation 수: 5 + 3 + 24 + 10 = 42
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (gameManager == null || gameManager.IsGameOver()) return;
        
        int roomChoice = actions.DiscreteActions[0]; // 0~7: 방 선택, 8: 현재 위치 유지
        int interactionChoice = actions.DiscreteActions[1]; // 0: 없음, 1: 부수기, 2: 고치기, 3: 사격
        
        // ===== 이동 처리 =====
        if (roomChoice < allRooms.Length && allRooms[roomChoice] != null)
        {
            InteractionPoint targetRoom = allRooms[roomChoice];
            npcController.MoveToRoom(targetRoom);
        }
        
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
        CalculatePeriodicReward();
    }
    
    void CalculatePeriodicReward()
    {
        if (SystemHealth.Instance == null) return;
        
        float currentHealth = SystemHealth.Instance.GetAverageHealth();
        float healthDelta = currentHealth - previousAverageHealth;
        
        if (isSaboteur)
        {
            // 안정도 떨어지면 +보상
            AddReward(-healthDelta * healthChangeRewardScale);
        }
        else
        {
            // 안정도 오르면 +보상
            AddReward(healthDelta * healthChangeRewardScale);
        }
        
        previousAverageHealth = currentHealth;
    }
    
    // ===== 이벤트 기반 보상 =====
    
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
        
        EndEpisode();
    }
    
    void TryShoot()
    {
        // 함장 총 로직 호출
        // CaptainGun 컴포넌트와 연동
        var captainGun = GetComponent<CaptainGun>();
        if (captainGun != null)
        {
            // 가장 가까운 캐릭터 타겟팅 (예시)
            // 실제로는 AI가 학습해서 결정하도록
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // 수동 테스트용 - 키보드 입력
        var discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = 8; // 기본: 이동 안함
        discreteActions[1] = 0; // 기본: 상호작용 안함
        
        // 숫자키 1~8로 방 선택
        for (int i = 0; i < 8; i++)
        {
            if (Input.GetKey(KeyCode.Alpha1 + i))
            {
                discreteActions[0] = i;
                break;
            }
        }
        
        // E: 수리, Q: 부수기, F: 사격
        if (Input.GetKey(KeyCode.E)) discreteActions[1] = 2;
        if (Input.GetKey(KeyCode.Q)) discreteActions[1] = 1;
        if (Input.GetKey(KeyCode.F)) discreteActions[1] = 3;
    }
}