using UnityEngine;

public class SystemHealth : MonoBehaviour
{
    // ===== 싱글톤 패턴 =====
    public static SystemHealth Instance { get; private set; }

    // ===== 전체 평균 안정도 =====
    [Header("=== 전체 평균 안정도 (읽기 전용) ===")]
    [SerializeField] private float averageHealth = 100f;
    [SerializeField] private bool shipStopped = false;

    // 모든 InteractionPoint 참조
    private InteractionPoint[] allRooms;

    // ===== 이벤트 =====
    public delegate void HealthChangedEvent(float averageHealth);
    public event HealthChangedEvent OnHealthChanged;

    public delegate void ShipStoppedEvent(bool stopped);
    public event ShipStoppedEvent OnShipStopped;

    public delegate void LocationAlertEvent(string roomName);
    public event LocationAlertEvent OnLocationAlert;

    // ===== Unity 생명주기 =====
    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        // 모든 InteractionPoint 찾기
        allRooms = FindObjectsOfType<InteractionPoint>();
        Debug.Log($"[시스템] {allRooms.Length}개 방 감지됨");
        UpdateAverageHealth();
    }

    void Update()
    {
        // 승리 조건 체크
        CheckWinConditions();
    }

    // ===== 전체 평균 계산 =====
    public void UpdateAverageHealth()
    {
        if (allRooms == null || allRooms.Length == 0) return;

        float total = 0f;
        foreach (var room in allRooms)
        {
            total += room.GetCurrentHealth();
        }
        averageHealth = total / allRooms.Length;

        Debug.Log($"[시스템] 전체 평균 안정도: {averageHealth:F1}%");

        // 이벤트 발생
        OnHealthChanged?.Invoke(averageHealth);

        // 50% 이하: 우주선 멈춤
        if (averageHealth <= GameManager.Instance.shipStopThreshold && !shipStopped)
        {
            shipStopped = true;
            GameManager.Instance.SetShipStopped(true);
            OnShipStopped?.Invoke(true);
            Debug.Log("⚠️ [경고] 평균 안정도 50% 이하! 우주선 멈춤!");
        }
        else if (averageHealth > GameManager.Instance.shipStopThreshold && shipStopped)
        {
            shipStopped = false;
            GameManager.Instance.SetShipStopped(false);
            OnShipStopped?.Invoke(false);
            Debug.Log("✓ [복구] 우주선 재가동!");
        }
    }

    // ===== 방 알림 (30% 이하) =====
    public void TriggerRoomAlert(string roomName)
    {
        Debug.Log($"🚨 [위치 공지] {roomName}에서 심각한 손상 감지!");
        OnLocationAlert?.Invoke(roomName);
    }

    // ===== 승리 조건 체크 =====
    void CheckWinConditions()
    {
        if (GameManager.Instance == null) return;
        if (GameManager.Instance.currentState != GameManager.GameState.Playing) return;

        // 사보타주 승리: 방 하나라도 0%
        foreach (var room in allRooms)
        {
            if (room.GetCurrentHealth() <= 0f)
            {
                GameManager.Instance.SabotageWin($"{room.GetRoomName()} 완전 파괴!");
                return;
            }
        }

        // 사보타주 승리: 전체 평균 30% 이하
        if (averageHealth <= GameManager.Instance.locationAlertThreshold)
        {
            GameManager.Instance.SabotageWin($"전체 평균 안정도 {averageHealth:F1}% (30% 이하)");
            return;
        }

        // 인간 승리: 목적지 도착
        if (GameManager.Instance.GetProgress() >= 1f)
        {
            GameManager.Instance.HumanWin("목적지 도착");
            return;
        }

        // 사보타주 승리: 인간과 동수
        int humanCount = GameManager.Instance.GetAliveHumanCount();
        int saboteurCount = GameManager.Instance.GetAliveSaboteurCount();

        if (humanCount <= saboteurCount && saboteurCount > 0)
        {
            GameManager.Instance.SabotageWin($"인간과 동수 ({humanCount} vs {saboteurCount})");
            return;
        }

        // 인간 승리: 사보타주 전멸
        if (saboteurCount == 0)
        {
            GameManager.Instance.HumanWin("모든 사보타주 제거");
            return;
        }
    }

    // ===== 시스템 리셋 =====

    public void ResetHealth()
    {
        UpdateAverageHealth();
        Debug.Log($"[시스템 리셋] 전체 평균 안정도: {averageHealth}%");
    }

    // ===== Getter =====
    public float GetAverageHealth() => averageHealth;
    public bool IsShipStopped() => shipStopped;
}