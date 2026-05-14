using UnityEngine;

public class PlayerController : MonoBehaviour
{
    // ===== 컴포넌트 참조 =====
    private CharacterController characterController;
    private Camera playerCamera;

    // ===== 이동 설정 =====
    [Header("=== 이동 설정 ===")]
    [Tooltip("GameManager 값 사용 시 체크 해제")]
    public bool useCustomSpeed = false;
    public float customMoveSpeed = 5f;

    [Header("=== 마우스 설정 ===")]
    public float mouseSensitivity = 2f;
    public float maxLookAngle = 80f;

    private float rotationX = 0f;

    // ===== 상호작용 설정 =====
    [Header("=== 상호작용 설정 ===")]
    public float interactionRange = 3f;
    public KeyCode interactionKey = KeyCode.E;

    // 현재 상호작용 가능한 오브젝트
    private InteractionPoint currentInteractionPoint;
    private bool isInteracting = false;
    private float interactionTimer = 0f;

    // ===== 플레이어 상태 =====
    [Header("=== 상태 (읽기 전용) ===")]
    [SerializeField] private bool canMove = true;
    [SerializeField] private bool isDead = false;

    // ===== Unity 생명주기 =====
    void Start()
    {
        // 컴포넌트 가져오기
        characterController = GetComponent<CharacterController>();
        if (characterController == null)
        {
            characterController = gameObject.AddComponent<CharacterController>();
        }

        // 카메라 찾기 (자식 또는 메인 카메라)
        playerCamera = GetComponentInChildren<Camera>();
        if (playerCamera == null)
        {
            playerCamera = Camera.main;
        }

        // 마우스 커서 숨기기 및 고정
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        // GameManager에 플레이어 등록
        if (GameManager.Instance != null)
        {
            GameManager.Instance.player = gameObject;
        }
    }

    void Update()
    {
        if (isDead) return;

        // 게임 상태 체크
        if (GameManager.Instance != null)
        {
            // Meeting 상태면 이동 불가
            if (GameManager.Instance.currentState == GameManager.GameState.Meeting)
            {
                canMove = false;
                HandleMouseLook();
                return;
            }
            else if (GameManager.Instance.currentState == GameManager.GameState.Playing)
            {
                canMove = true;
            }
        }

        if (canMove && !isInteracting)
        {
            HandleMovement();
            HandleMouseLook();
        }

        HandleInteraction();
        CheckForInteractionPoint();

        // ESC로 마우스 커서 토글 (디버그용)
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ToggleCursor();
        }
    }

    // ===== 이동 처리 =====
    void HandleMovement()
    {
        float moveSpeed = useCustomSpeed ? customMoveSpeed : GameManager.Instance.moveSpeed;

        float horizontal = Input.GetAxis("Horizontal"); // A, D
        float vertical = Input.GetAxis("Vertical");     // W, S

        Vector3 moveDirection = transform.right * horizontal + transform.forward * vertical;
        moveDirection = moveDirection.normalized * moveSpeed;

        // 중력 적용
        if (!characterController.isGrounded)
        {
            moveDirection.y = -9.8f;
        }

        characterController.Move(moveDirection * Time.deltaTime);
    }

    // ===== 마우스 시점 처리 =====
    void HandleMouseLook()
    {
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

        // 좌우 회전 (캐릭터 전체)
        transform.Rotate(Vector3.up * mouseX);

        // 상하 회전 (카메라만)
        rotationX -= mouseY;
        rotationX = Mathf.Clamp(rotationX, -maxLookAngle, maxLookAngle);

        if (playerCamera != null)
        {
            playerCamera.transform.localRotation = Quaternion.Euler(rotationX, 0f, 0f);
        }
    }

    // ===== 상호작용 포인트 감지 =====
    void CheckForInteractionPoint()
    {
        if (playerCamera == null) return;

        Ray ray = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, interactionRange))
        {
            InteractionPoint point = hit.collider.GetComponent<InteractionPoint>();
            
            if (point != null)
            {
                currentInteractionPoint = point;
                // UI 힌트 표시 가능 (예: "E키로 상호작용")
            }
            else
            {
                currentInteractionPoint = null;
            }
        }
        else
        {
            currentInteractionPoint = null;
        }
    }

    // ===== 상호작용 처리 =====
    void HandleInteraction()
    {
        // 상호작용 시작
        if (Input.GetKeyDown(interactionKey) && currentInteractionPoint != null && !isInteracting)
        {
            StartInteraction();
        }

        // 상호작용 진행 중
        if (isInteracting)
        {
            if (Input.GetKey(interactionKey))
            {
                interactionTimer += Time.deltaTime;

                // 상호작용 완료 체크
                float requiredTime = GetInteractionTime();
                if (interactionTimer >= requiredTime)
                {
                    CompleteInteraction();
                }
            }
            else
            {
                // 키를 떼면 취소
                CancelInteraction();
            }
        }
    }

    void StartInteraction()
    {
        isInteracting = true;
        interactionTimer = 0f;
        canMove = false;

        Debug.Log($"[상호작용 시작] {currentInteractionPoint.roomName}");
    }

    void CompleteInteraction()
    {
        if (currentInteractionPoint != null)
        {
            // 역할에 따라 부수기/고치기 결정
            bool isSaboteur = RoleManager.Instance.IsSaboteur(gameObject);
            currentInteractionPoint.OnInteractionComplete(gameObject, isSaboteur);
        }

        isInteracting = false;
        interactionTimer = 0f;
        canMove = true;

        Debug.Log("[상호작용 완료]");
    }

    void CancelInteraction()
    {
        isInteracting = false;
        interactionTimer = 0f;
        canMove = true;

        Debug.Log("[상호작용 취소]");
    }

    float GetInteractionTime()
    {
        if (GameManager.Instance == null) return 3f;

        bool isSaboteur = RoleManager.Instance.IsSaboteur(gameObject);
        return isSaboteur ? GameManager.Instance.sabotageTime : GameManager.Instance.repairTime;
    }

    // ===== 상호작용 진행률 (UI용) =====
    public float GetInteractionProgress()
    {
        if (!isInteracting) return 0f;
        float requiredTime = GetInteractionTime();
        return interactionTimer / requiredTime;
    }

    // ===== 카페로 강제 이동 (함장 소집 시) =====
    public void TeleportToCafe()
    {
        if (GameManager.Instance.cafePosition != null)
        {
            // CharacterController는 직접 position 변경이 안되므로 비활성화 후 이동
            characterController.enabled = false;
            transform.position = GameManager.Instance.cafePosition.position;
            characterController.enabled = true;

            Debug.Log($"[소집] {gameObject.name}이(가) 카페로 이동했습니다.");
        }

        // 진행 중인 상호작용 취소
        CancelInteraction();
    }

    // ===== 사망 처리 =====
    public void Die()
    {
        isDead = true;
        canMove = false;

        Debug.Log($"[사망] {gameObject.name}");

        // GameManager에서 제거
        GameManager.Instance.RemoveCharacter(gameObject);

        // 오브젝트 비활성화 또는 사망 애니메이션
        gameObject.SetActive(false);
    }

    // ===== 유틸리티 =====
    void ToggleCursor()
    {
        if (Cursor.lockState == CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
        else
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }

    // ===== Getter =====
    public bool IsDead() => isDead;
    public bool IsInteracting() => isInteracting;
    public InteractionPoint GetCurrentInteractionPoint() => currentInteractionPoint;
}