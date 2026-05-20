using UnityEngine;
using System.Collections.Generic;

public class CaptainGun : MonoBehaviour
{
    // ===== 총알 상태 =====
    [Header("=== 총알 상태 (읽기 전용) ===")]
    [SerializeField] private int remainingBullets;

    // ===== 소집 상태 =====
    [Header("=== 소집 상태 ===")]
    [SerializeField] private bool isMeetingActive = false;

    // ===== 조준선 UI =====
    [Header("=== 조준선 설정 ===")]
    public GameObject crosshairUI;  // 조준선 오브젝트

    // ===== 소집된 캐릭터 위치 =====
    [Header("=== 카페 배치 설정 ===")]
    public float circleRadius = 3f;  // 원형 배치 반경
    public Transform cafeCenter;      // 카페 중심 (없으면 GameManager 것 사용)

    // ===== 참조 =====
    private PlayerController playerController;
    private bool isCaptain = false;

    // ===== 이벤트 =====
    public delegate void MeetingEvent();
    public event MeetingEvent OnMeetingStart;
    public event MeetingEvent OnMeetingEnd;

    public delegate void ExecutionEvent(GameObject target, bool wasSaboteur);
    public event ExecutionEvent OnExecution;

    // ===== Unity 생명주기 =====
    void Start()
    {
        playerController = GetComponent<PlayerController>();

        // 총알 초기화
        if (GameManager.Instance != null)
        {
            remainingBullets = GameManager.Instance.captainBullets;
        }
        else
        {
            remainingBullets = 2;
        }

        if (crosshairUI != null)
        {
            crosshairUI.SetActive(false);
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            Debug.Log($"[CaptainGun] Q키 감지 - isCaptain: {isCaptain}");
        }

        // 함장인지 확인
        if (RoleManager.Instance != null)
        {
            isCaptain = RoleManager.Instance.IsCaptain(gameObject);
        }

        if (!isCaptain) return;

        // 게임 상태 체크
        if (GameManager.Instance.currentState != GameManager.GameState.Playing &&
            GameManager.Instance.currentState != GameManager.GameState.Meeting)
        {
            return;
        }

        // 입력 처리
        HandleInput();
    }

    // ===== 입력 처리 =====
    void HandleInput()
    {
        // Q키: 카페 소집
        if (Input.GetKeyDown(KeyCode.Q) && !isMeetingActive && remainingBullets > 0)
        {
            StartMeeting();
        }

        // Meeting 상태에서 마우스 클릭: 타겟 선택 및 처형
        if (isMeetingActive && Input.GetMouseButtonDown(0))
        {
            TrySelectAndExecute();
        }

        // ESC: 소집 취소 (처형 안하고 끝내기)
        if (isMeetingActive && Input.GetKeyDown(KeyCode.Escape))
        {
            CancelMeeting();
        }
    }

    // ===== 카페 소집 =====
    void StartMeeting()
    {
        if (remainingBullets <= 0)
        {
            Debug.Log("[함장] 총알이 없습니다!");
            return;
        }

        isMeetingActive = true;
        GameManager.Instance.SetGameState(GameManager.GameState.Meeting);

        // 조준선 표시
        if (crosshairUI != null)
        {
            crosshairUI.SetActive(true);
        }

        // 마우스 커서 숨기기
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        Debug.Log("📢 [함장] 전체 소집! 모두 카페로 모이세요!");

        TeleportAllToCafe();
        OnMeetingStart?.Invoke();
    }

    // ===== 모든 캐릭터 카페로 이동 =====
    void TeleportAllToCafe()
    {
        List<GameObject> allCharacters = GameManager.Instance.allCharacters;
        Transform center = cafeCenter != null ? cafeCenter : GameManager.Instance.cafePosition;

        if (center == null)
        {
            Debug.LogError("카페 위치가 설정되지 않았습니다!");
            return;
        }

        int count = allCharacters.Count;
        float angleStep = 360f / count;

        for (int i = 0; i < count; i++)
        {
            GameObject character = allCharacters[i];
            if (character == null || !character.activeInHierarchy) continue;

            // 원형으로 배치
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 offset = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)) * circleRadius;
            Vector3 targetPosition = center.position + offset;

            // 플레이어인지 NPC인지 확인
            PlayerController pc = character.GetComponent<PlayerController>();
            NPCController npc = character.GetComponent<NPCController>();

            if (pc != null)
            {
                pc.TeleportToCafe();
                // 위치 미세 조정
                character.transform.position = targetPosition;
            }
            else if (npc != null)
            {
                npc.TeleportToCafeInstant();
                character.transform.position = targetPosition;
            }

            // 중앙을 바라보게
            character.transform.LookAt(new Vector3(center.position.x, character.transform.position.y, center.position.z));
        }

        Debug.Log($"[소집 완료] {count}명이 카페에 모였습니다.");
    }

    // ===== 타겟 선택 및 처형 =====
    void TrySelectAndExecute()
    {
        // Raycast로 타겟 선택
        Camera cam = Camera.main;
        if (cam == null) return;

        Ray ray = cam.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, 50f))
        {
            GameObject target = hit.collider.gameObject;

            // 자기 자신은 선택 불가
            if (target == gameObject)
            {
                Debug.Log("[함장] 자기 자신은 처형할 수 없습니다.");
                return;
            }

            // 유효한 캐릭터인지 확인
            if (GameManager.Instance.allCharacters.Contains(target))
            {
                ExecuteTarget(target);
            }
            else
            {
                // 부모 오브젝트 확인 (콜라이더가 자식에 있을 경우)
                Transform parent = target.transform.parent;
                while (parent != null)
                {
                    if (GameManager.Instance.allCharacters.Contains(parent.gameObject))
                    {
                        ExecuteTarget(parent.gameObject);
                        return;
                    }
                    parent = parent.parent;
                }

                Debug.Log("[함장] 유효한 대상이 아닙니다.");
            }
        }
    }

    // ===== 처형 실행 =====
    void ExecuteTarget(GameObject target)
    {
        if (target == null) return;

        // 총알 소모
        remainingBullets--;

        // 역할 확인 (정체는 공개하지 않음!)
        bool wasSaboteur = RoleManager.Instance.IsSaboteur(target);

        Debug.Log($"🔫 [처형] {target.name}이(가) 처형되었습니다.");
        Debug.Log($"[함장] 남은 총알: {remainingBullets}발");

        // 타겟 사망 처리
        PlayerController pc = target.GetComponent<PlayerController>();
        NPCController npc = target.GetComponent<NPCController>();

        if (pc != null)
        {
            pc.Die();
        }
        else if (npc != null)
        {
            npc.Die();
        }

        // 이벤트 발생 (UI에서 사용, 정체는 비공개)
        OnExecution?.Invoke(target, wasSaboteur);

        // 소집 종료
        EndMeeting();

        // 승리 조건 즉시 체크
        CheckWinAfterExecution(wasSaboteur);
    }

    // ===== 처형 후 승리 조건 체크 =====
    void CheckWinAfterExecution(bool wasSaboteur)
    {
        int humanCount = GameManager.Instance.GetAliveHumanCount();
        int saboteurCount = GameManager.Instance.GetAliveSaboteurCount();

        if (wasSaboteur)
        {
            // 사보타주 죽임
            if (saboteurCount == 0)
            {
                GameManager.Instance.HumanWin("모든 사보타주 제거");
            }
            else
            {
                Debug.Log($"[상황] 사보타주 {saboteurCount}명 남음");
            }
        }
        else
        {
            // 인간 죽임 (실수)
            if (humanCount <= saboteurCount)
            {
                GameManager.Instance.SabotageWin($"인간과 동수 ({humanCount} vs {saboteurCount})");
            }
            else
            {
                Debug.Log($"[상황] 인간 {humanCount}명, 사보타주 {saboteurCount}명 남음");
            }
        }
    }

    // ===== 소집 취소 =====
    void CancelMeeting()
    {
        Debug.Log("[함장] 소집을 취소합니다.");
        EndMeeting();
    }

    // ===== 소집 종료 =====
    void EndMeeting()
    {
        isMeetingActive = false;
        GameManager.Instance.SetGameState(GameManager.GameState.Playing);

        // 조준선 숨기기
        if (crosshairUI != null)
        {
            crosshairUI.SetActive(false);
        }

        Debug.Log("[소집 종료] 게임을 계속합니다.");
        OnMeetingEnd?.Invoke();
    }

    // ===== Getter =====
    public int GetRemainingBullets() => remainingBullets;
    public bool IsMeetingActive() => isMeetingActive;
    public bool IsCaptain() => isCaptain;

    // ===== 함장 여부 확인 후 컴포넌트 활성화 =====
    public void EnableCaptainAbility()
    {
        isCaptain = true;
        Debug.Log($"[함장] {gameObject.name}이(가) 함장 능력을 얻었습니다. 총알: {remainingBullets}발");
    }

    // ===== ML-Agents 전용: 직접 사격 (회의 없이) =====
    public bool TryExecuteTarget(GameObject target)
    {
        if (target == null || remainingBullets <= 0) return false;

        remainingBullets--;

        bool wasSaboteur = RoleManager.Instance != null && RoleManager.Instance.IsSaboteur(target);

        NPCController npc = target.GetComponent<NPCController>();
        if (npc != null) npc.Die();

        OnExecution?.Invoke(target, wasSaboteur);

        CheckWinAfterExecution(wasSaboteur);
        return true;
    }

    // ===== 총알/상태 리셋 (에피소드 리셋용) =====
    public void ResetGun()
    {
        remainingBullets = GameManager.Instance != null ? GameManager.Instance.captainBullets : 2;
        isMeetingActive = false;
    }
}

