using UnityEngine;

/// <summary>
/// 관전용 자유 카메라.
/// - 우클릭 누른 채 마우스 이동: 시점 회전
/// - WASD: 수평 이동 / Q·E: 하강·상승
/// - Shift: 가속 / 마우스 휠: 이동 속도 조절
///
/// 사용법:
///   1) 씬에 빈 GameObject 생성 → Camera 컴포넌트 추가 (또는 기존 Main Camera 사용)
///   2) 이 스크립트를 그 카메라에 붙임
///   3) 관전용으로 빌드 (학습용 빌드와 다른 폴더에!)
/// </summary>
public class SpectatorCamera : MonoBehaviour
{
    [Header("이동")]
    public float moveSpeed = 10f;
    public float fastMultiplier = 3f;    // Shift 누를 때 배속
    public float scrollSpeedStep = 2f;   // 휠로 속도 조절 단위
    public float minSpeed = 2f;
    public float maxSpeed = 100f;

    [Header("회전")]
    public float lookSensitivity = 3f;
    public bool invertY = false;

    private float yaw;
    private float pitch;

    void Start()
    {
        Vector3 e = transform.eulerAngles;
        yaw = e.y;
        pitch = e.x;
    }

    void Update()
    {
        HandleLook();
        HandleMove();
        HandleSpeedScroll();
    }

    void HandleLook()
    {
        // 우클릭을 누르고 있을 때만 회전 (녹화 중 실수 방지)
        if (!Input.GetMouseButton(1)) return;

        float mx = Input.GetAxis("Mouse X") * lookSensitivity;
        float my = Input.GetAxis("Mouse Y") * lookSensitivity * (invertY ? 1f : -1f);

        yaw += mx;
        pitch += my;
        pitch = Mathf.Clamp(pitch, -89f, 89f);

        transform.eulerAngles = new Vector3(pitch, yaw, 0f);
    }

    void HandleMove()
    {
        float speed = moveSpeed;
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
            speed *= fastMultiplier;

        Vector3 dir = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) dir += transform.forward;
        if (Input.GetKey(KeyCode.S)) dir -= transform.forward;
        if (Input.GetKey(KeyCode.D)) dir += transform.right;
        if (Input.GetKey(KeyCode.A)) dir -= transform.right;
        if (Input.GetKey(KeyCode.E)) dir += Vector3.up;
        if (Input.GetKey(KeyCode.Q)) dir -= Vector3.up;

        transform.position += dir.normalized * speed * Time.unscaledDeltaTime;
    }

    void HandleSpeedScroll()
    {
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.001f)
        {
            moveSpeed = Mathf.Clamp(moveSpeed + scroll * scrollSpeedStep * 10f, minSpeed, maxSpeed);
        }
    }
}