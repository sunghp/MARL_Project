using UnityEngine;

public class InteractionPoint : MonoBehaviour
{
    // ===== 방 정보 =====
    [Header("=== 방 정보 ===")]
    public string roomName = "Room";
    public int roomIndex = 0;

    // ===== 개별 안정도 =====
    [Header("=== 방 안정도 ===")]
    public float maxHealth = 100f;
    [SerializeField] private float currentHealth;
    [SerializeField] private bool hasAlerted = false;  // 30% 알림 여부

    // ===== 상태 =====
    [Header("=== 상태 (읽기 전용) ===")]
    [SerializeField] private bool isBeingUsed = false;
    [SerializeField] private GameObject currentUser = null;

    // ===== 이벤트 (선택적) =====
    public delegate void InteractionEvent(GameObject user, bool isSabotage);
    public event InteractionEvent OnInteraction;

    // ===== 사보타주 감지 이벤트 (함장 AI용) =====
    public delegate void SabotageDetectedEvent(GameObject saboteur, string roomName, Vector3 position);
    public static event SabotageDetectedEvent OnSabotageDetected;

    // ===== 수리 완료 이벤트 (NPC AI 알람 기억 해제용) =====
    public delegate void RoomRepairedEvent(string roomName);
    public static event RoomRepairedEvent OnRoomRepaired;

    void Start()
    {
        currentHealth = maxHealth;
    }

    // ===== 상호작용 완료 처리 =====
    public void OnInteractionComplete(GameObject user, bool isSaboteur)
    {
        if (SystemHealth.Instance == null)
        {
            Debug.LogError("SystemHealth가 없습니다!");
            return;
        }

        if (isSaboteur)
        {
            // 사보타주: 부수기
            float damage = GameManager.Instance.sabotageDamage;
            currentHealth = Mathf.Max(0f, currentHealth - damage);
#if UNITY_EDITOR
            Debug.Log($"[부수기] {user.name}이(가) {roomName}을(를) 파괴! (안정도: {currentHealth}%)");
#endif

            // 사보타주 감지 이벤트 발생 (함장 AI가 구독)
            OnSabotageDetected?.Invoke(user, roomName, transform.position);

            // 30% 이하 알림
            if (currentHealth <= 30f && !hasAlerted)
            {
                hasAlerted = true;
                SystemHealth.Instance.TriggerRoomAlert(roomName);
            }
        }
        else
        {
            // 인간: 고치기
            float repair = GameManager.Instance.repairAmount;
            float prevHealth = currentHealth;
            currentHealth = Mathf.Min(maxHealth, currentHealth + repair);
#if UNITY_EDITOR
            Debug.Log($"[고치기] {user.name}이(가) {roomName}을(를) 수리! (안정도: {currentHealth}%)");
#endif

            // 30% 초과하면 알림 리셋 + 수리 완료 이벤트
            if (currentHealth > 30f && hasAlerted)
            {
                hasAlerted = false;
                OnRoomRepaired?.Invoke(roomName);
                Debug.Log($"✅ [수리 완료] {roomName} 알람 해제!");
            }
        }

        // SystemHealth에 전체 평균 업데이트 요청
        SystemHealth.Instance.UpdateAverageHealth();

        // 이벤트 발생
        OnInteraction?.Invoke(user, isSaboteur);

        // 사용 상태 해제
        isBeingUsed = false;
        currentUser = null;
    }

    // ===== 상호작용 시작 =====
    public bool TryStartInteraction(GameObject user)
    {
        if (isBeingUsed)
        {
            Debug.Log($"[상호작용 불가] {roomName}은(는) 이미 사용 중입니다.");
            return false;
        }

        isBeingUsed = true;
        currentUser = user;
        return true;
    }

    // ===== 상호작용 취소 =====
    public void CancelInteraction(GameObject user)
    {
        if (currentUser == user)
        {
            isBeingUsed = false;
            currentUser = null;
            Debug.Log($"[상호작용 취소] {user.name}이(가) {roomName} 상호작용을 취소했습니다.");
        }
    }

    // ===== Getter =====
    public float GetCurrentHealth() => currentHealth;
    public float GetHealthPercent() => currentHealth / maxHealth;
    public bool IsBeingUsed() => isBeingUsed;
    public GameObject GetCurrentUser() => currentUser;
    public string GetRoomName() => roomName;

    // ===== 시각적 표시 (에디터용) =====
    void OnDrawGizmos()
    {
        Gizmos.color = isBeingUsed ? Color.red : Color.green;
        Gizmos.DrawWireSphere(transform.position, 0.5f);
    }

    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, 2f);
    }

    // ===== 방 리셋 =====

    public void ResetRoom()
    {
        currentHealth = maxHealth;
        isBeingUsed = false;
        currentUser = null;
        hasAlerted = false; // 알람 플래그도 리셋
        
#if UNITY_EDITOR
        Debug.Log($"[리셋] {roomName} 안정도: {currentHealth}%");
#endif
}
}