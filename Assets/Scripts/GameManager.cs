using UnityEngine;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.AI.Navigation;

public class GameManager : MonoBehaviour
{
    // ===== 싱글톤 패턴 =====
    public static GameManager Instance { get; private set; }

    // ===== 게임 상태 =====
    public enum GameState
    {
        Preparing,      // 게임 시작 전
        Playing,        // 게임 진행 중
        Meeting,        // 카페 소집 (함장 처형 시간)
        HumanWin,       // 인간 승리
        SabotageWin     // 사보타주 승리
    }
    
    [Header("=== 현재 게임 상태 ===")]
    public GameState currentState = GameState.Preparing;

    // ===== 밸런스 파라미터 - Inspector에서 조정 가능 =====
    [Header("=== 항해 설정 ===")]
    [Tooltip("목적지까지의 거리")]
    public float totalDistance = 1000f;

    [Tooltip("우주선 항해 속도 (단위/초)")]
    public float shipSpeed = 10f;

    [Tooltip("우주선 멈춤 시 속도")]
    public float stoppedSpeed = 0f;

    [Header("=== 시스템 안정도 파라미터 ===")]
    [Tooltip("시작 안정도")]
    public float maxSystemHealth = 100f;

    [Tooltip("우주선 멈추는 임계값")]
    public float shipStopThreshold = 50f;

    [Tooltip("위치 공지 임계값")]
    public float locationAlertThreshold = 30f;

    [Tooltip("사보타주 승리 임계값 (전체 평균 안정도)")]
    public float sabotageWinThreshold = 30f;

    [Header("=== 상호작용 파라미터 ===")]
    [Tooltip("부수기 시간 (초)")]
    public float sabotageTime = 3f;

    [Tooltip("부수기 시 감소량")]
    public float sabotageDamage = 10f;

    [Tooltip("고치기 시간 (초)")]
    public float repairTime = 5f;

    [Tooltip("고치기 시 증가량")]
    public float repairAmount = 15f;

    [Header("=== 이동/시야 파라미터 ===")]
    [Tooltip("이동 속도")]
    public float moveSpeed = 5f;

    [Tooltip("시야 거리")]
    public float visionRange = 10f;

    [Header("=== 역할 파라미터 ===")]
    [Tooltip("사보타주 인원 수")]
    public int sabotageCount = 2;

    [Tooltip("일반 인간 인원 수 (함장 제외)")]
    public int humanCount = 3;

    [Header("=== 함장 파라미터 ===")]
    [Tooltip("함장 총알 개수")]
    public int captainBullets = 2;

    [Header("=== 주요 위치 ===")]
    [Tooltip("카페 소집 위치")]
    public Transform cafePosition;

    // ===== 항해 런타임 변수 =====
    [Header("=== 항해 상태 (읽기 전용) ===")]
    [SerializeField] private float currentDistance = 0f;
    [SerializeField] private bool shipStopped = false;

    // ===== 캐릭터 리스트 =====
    [Header("=== 캐릭터 리스트 ===")]
    public GameObject player;
    public GameObject captain;
    public List<GameObject> allCharacters = new List<GameObject>();
    public List<GameObject> saboteurs = new List<GameObject>();
    public List<GameObject> humans = new List<GameObject>(); // 함장 제외 일반 인간

    [Header("=== 게임 상태 ===")]
    private bool isGameOver = false;

    // ===== ML-Agents 에피소드 관리 =====
    private List<NPCAgent> allNPCAgents = new List<NPCAgent>();
    private List<GameObject> allCharactersOriginal = new List<GameObject>(); // 원본 (사망해도 유지)

    [Header("=== 에피소드 시간 제한 ===")]
    [Tooltip("최대 에피소드 시간 (초). 초과 시 무승부→사보타주 승리 처리")]
    public float maxEpisodeTime = 300f;
    private float episodeTimer = 0f;

    // ===== ML-Agents 헬퍼 =====

    public bool IsGameOver()
    {
        return isGameOver;
    }

    public float GetDistanceProgress()
    {
        return currentDistance / totalDistance;
    }

    public float GetAliveHumanRatio()
    {
        if (RoleManager.Instance == null) return 0f;

        int aliveHumans = 0;
        int totalHumans = 0;

        // 원본 리스트 기준으로 계산 (사망자 포함)
        var sourceList = allCharactersOriginal.Count > 0 ? allCharactersOriginal : allCharacters;
        foreach (var character in sourceList)
        {
            if (character == null) continue;
            if (!RoleManager.Instance.IsSaboteur(character))
            {
                totalHumans++;
                if (character.activeInHierarchy)
                    aliveHumans++;
            }
        }

        return totalHumans > 0 ? (float)aliveHumans / totalHumans : 0f;
    }

    // 게임 종료 시 모든 에이전트에게 알림 + 동기화 리셋
    public void NotifyGameEnd(bool humanWin)
    {
        // 1. 죽은 에이전트 재활성화 (EndEpisode를 받을 수 있도록)
        foreach (var agent in allNPCAgents)
        {
            if (agent != null && !agent.gameObject.activeInHierarchy)
                agent.gameObject.SetActive(true);
        }

        // 2. 보상 부여 (아직 EndEpisode 안함)
        foreach (var agent in allNPCAgents)
        {
            if (agent != null)
                agent.OnGameEnd(humanWin);
        }

        // 3. 게임 상태 리셋 (새 에피소드 준비)
        ResetGame();

        // 4. EndEpisode 일괄 호출 → OnEpisodeBegin이 리셋된 상태로 실행됨
        foreach (var agent in allNPCAgents)
        {
            if (agent != null)
                agent.EndEpisode();
        }
    }

    // ===== Unity 생명주기 =====
    void Awake()
    {
        var surface = FindObjectOfType<NavMeshSurface>();
        if (surface != null)
        {
            surface.BuildNavMesh();
            Debug.Log("[NavMesh] 런타임 베이크 완료");
        }
        if (Instance == null)
        {
            Instance = this;
            // DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        InitializeGame();
    }

    void Update()
    {
        if (currentState == GameState.Playing)
        {
            UpdateShipProgress();

            // 에피소드 시간 제한 (무한 에피소드 방지)
            episodeTimer += Time.deltaTime;
            if (episodeTimer >= maxEpisodeTime)
            {
                SabotageWin("에피소드 시간 초과");
            }
        }
    }

    // ===== 게임 초기화 =====
    void InitializeGame()
    {
        currentDistance = 0f;
        shipStopped = false;
        currentState = GameState.Playing;

        Debug.Log("========== 게임 시작 ==========");
        Debug.Log($"목적지: {totalDistance} / 속도: {shipSpeed}");

        // 리스트 초기화
        allCharacters.Clear();
        saboteurs.Clear();
        humans.Clear();
        captain = null;

        // Player 찾기 (Tag: Player)
        player = GameObject.FindWithTag("Player");
        if (player != null) 
        {
            allCharacters.Add(player);
        }

        // NPC 찾기 (Tag: NPC)
        GameObject[] npcs = GameObject.FindGameObjectsWithTag("NPC");
        foreach (var npc in npcs)
        {
            allCharacters.Add(npc);
        }

        Debug.Log($"캐릭터 수집 완료: {allCharacters.Count}명");

        // ML-Agents: NPCAgent 참조 수집 (사망해도 유지, 에피소드 관리용)
        allNPCAgents.Clear();
        foreach (var character in allCharacters)
        {
            NPCAgent npcAgent = character.GetComponent<NPCAgent>();
            if (npcAgent != null)
                allNPCAgents.Add(npcAgent);
        }
        Debug.Log($"[ML-Agents] NPCAgent {allNPCAgents.Count}개 등록");

        // 원본 캐릭터 리스트 보존 (에피소드 리셋 시 복원용)
        allCharactersOriginal = new List<GameObject>(allCharacters);

        // 역할 배정
        if (RoleManager.Instance != null && allCharacters.Count > 0)
        {
            RoleManager.Instance.AssignRoles(allCharacters);
            RoleManager.Instance.AnnounceCaptain();
        }
    }

    // ===== 항해 진행 =====
    void UpdateShipProgress()
    {
        float currentSpeed = shipStopped ? stoppedSpeed : shipSpeed;
        currentDistance += currentSpeed * Time.deltaTime;
    }

    // ===== 게임 상태 변경 (다른 스크립트에서 호출) =====
    public void SetGameState(GameState newState)
    {
        currentState = newState;
        Debug.Log($"[게임 상태 변경] → {newState}");
    }

    public void SetShipStopped(bool stopped)
    {
        shipStopped = stopped;
        Debug.Log(stopped ? "⚠️ 우주선 정지!" : "✓ 우주선 재가동!");
    }

    // ===== 승리 처리 =====
    public void HumanWin(string reason)
    {
        if (isGameOver) return;  // 중복 호출 방지
        isGameOver = true;
        currentState = GameState.HumanWin;
        Debug.Log($"========== 인간 승리! ==========");
        Debug.Log($"이유: {reason}");
        NotifyGameEnd(true);
    }

    public void SabotageWin(string reason)
    {
        if (isGameOver) return;  // 중복 호출 방지
        isGameOver = true;
        currentState = GameState.SabotageWin;
        Debug.Log($"========== 사보타주 승리! ==========");
        Debug.Log($"이유: {reason}");
        NotifyGameEnd(false);
    }

    // ===== 캐릭터 관리 =====
    public void RegisterCharacter(GameObject character, bool isSaboteur, bool isCaptain)
    {
        if (isCaptain)
        {
            captain = character;
        }
        else if (isSaboteur)
        {
            saboteurs.Add(character);
        }
        else
        {
            humans.Add(character);
        }
#if UNITY_EDITOR
        string role = isCaptain ? "함장" : (isSaboteur ? "사보타주" : "인간");
        Debug.Log($"[등록] {role}: {character.name}");
#endif
    }

    public void RemoveCharacter(GameObject character)
    {
        // 함장은 죽을 수 없음
        if (character == captain)
        {
            Debug.LogWarning("함장은 제거할 수 없습니다!");
            return;
        }

        allCharacters.Remove(character);
        saboteurs.Remove(character);
        humans.Remove(character);

#if UNITY_EDITOR
        Debug.Log($"[제거] {character.name}");
#endif
    }

    // ===== Getter 메서드 =====
    public float GetCurrentDistance() => currentDistance;
    public float GetProgress() => currentDistance / totalDistance;
    public bool IsShipStopped() => shipStopped;

    public int GetAliveHumanCount()
    {
        // 함장 + 일반 인간
        int count = humans.Count;
        if (captain != null) count++;
        return count;
    }

    public int GetAliveSaboteurCount()
    {
        return saboteurs.Count;
    }

    // ===== 게임 리셋 (ML 학습용) =====

    public void ResetGame()
    {
        Debug.Log("========== 게임 리셋 ==========");

        // 0. ML-Agents environment_parameters에서 밸런스 파라미터 읽기
        LoadEnvironmentParameters();

        // 1. 게임 상태 초기화 + 캐릭터 리스트 복원
        // isGameOver는 리셋 완료 후 마지막에 false로 (리셋 중 CheckWinConditions 오발 방지)
        currentState = GameState.Preparing;
        currentDistance = 0f;
        shipStopped = false;
        episodeTimer = 0f;

        // 원본에서 allCharacters 복원 (RemoveCharacter로 빠진 캐릭터 복구)
        allCharacters = new List<GameObject>(allCharactersOriginal);
        saboteurs.Clear();
        humans.Clear();
        captain = null;
        
        // 2. 모든 InteractionPoint 안정도 초기화
        InteractionPoint[] allRooms = FindObjectsOfType<InteractionPoint>();
        foreach (var room in allRooms)
        {
            room.ResetRoom();
        }
        
        // 3. SystemHealth 초기화
        if (SystemHealth.Instance != null)
        {
            SystemHealth.Instance.ResetHealth();
        }
        
        // 4. 모든 캐릭터 활성화 및 위치 초기화
        foreach (var character in allCharacters)
        {
            if (character != null)
            {
                character.SetActive(true);
                
                // 카페 위치로 이동 (랜덤 오프셋)
                Vector3 spawnPos = cafeSpawnPoint != null ? 
                    cafeSpawnPoint.position + new Vector3(Random.Range(-2f, 2f), 0, Random.Range(-2f, 2f)) :
                    Vector3.zero;
                
                character.transform.position = spawnPos;
                
                // NavMeshAgent 리셋
                UnityEngine.AI.NavMeshAgent agent = character.GetComponent<UnityEngine.AI.NavMeshAgent>();
                if (agent != null)
                {
                    agent.ResetPath();
                    agent.Warp(spawnPos);
                }
                
                // NPCController 리셋
                NPCController npcController = character.GetComponent<NPCController>();
                if (npcController != null)
                {
                    npcController.ResetNPC();
                }
                
                // NPCAIBrain 리셋
                NPCAIBrain brain = character.GetComponent<NPCAIBrain>();
                if (brain != null)
                {
                    brain.ResetBrain();
                }

                // CaptainGun 리셋 (이전 에피소드 함장의 잔여 상태 초기화)
                CaptainGun gun = character.GetComponent<CaptainGun>();
                if (gun != null)
                {
                    gun.ResetGun();
                }
            }
        }
        
        // 5. 역할 재배정
        if (RoleManager.Instance != null)
        {
            RoleManager.Instance.ResetRoles();
            RoleManager.Instance.AssignRoles(allCharacters);
        }

        // 6. 모든 초기화 완료 후 게임 시작 (이 시점에서야 CheckWinConditions 허용)
        isGameOver = false;
        currentState = GameState.Playing;

        Debug.Log("========== 게임 시작 ==========");
    }

    [Header("스폰 설정")]
    public Transform cafeSpawnPoint; // Inspector에서 카페 위치 할당

    // ===== ML-Agents 밸런스 파라미터 연동 =====
    void LoadEnvironmentParameters()
    {
        var envParams = Academy.Instance.EnvironmentParameters;

        totalDistance       = envParams.GetWithDefault("total_distance", totalDistance);
        shipSpeed           = envParams.GetWithDefault("ship_speed", shipSpeed);
        shipStopThreshold   = envParams.GetWithDefault("ship_stop_threshold", shipStopThreshold);
        sabotageWinThreshold = envParams.GetWithDefault("sabotage_win_threshold", sabotageWinThreshold);
        sabotageDamage      = envParams.GetWithDefault("sabotage_damage", sabotageDamage);
        repairAmount        = envParams.GetWithDefault("repair_amount", repairAmount);
        sabotageTime        = envParams.GetWithDefault("sabotage_time", sabotageTime);
        repairTime          = envParams.GetWithDefault("repair_time", repairTime);
        moveSpeed           = envParams.GetWithDefault("move_speed", moveSpeed);
        visionRange         = envParams.GetWithDefault("vision_range", visionRange);
        maxEpisodeTime      = envParams.GetWithDefault("max_episode_time", maxEpisodeTime);
    }
}