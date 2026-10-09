// 컴파일 검사 전용 최소 스텁 (실제 동작 없음). Unity/ML-Agents API 시그니처만 흉내낸다.
using System;
using System.Collections.Generic;

namespace UnityEngine
{
    public class Object {
        public string name;
        public static void Destroy(Object o) {}
        public static T FindObjectOfType<T>() where T : Object { return null; }
        public static T[] FindObjectsOfType<T>() where T : Object { return new T[0]; }
        public static implicit operator bool(Object o) { return !ReferenceEquals(o, null); }
    }
    public class Component : Object {
        public GameObject gameObject; public Transform transform; public string tag;
        public T GetComponent<T>() { return default(T); }
        public T GetComponentInChildren<T>() { return default(T); }
        public bool CompareTag(string t) { return false; }
    }
    public class Behaviour : Component { public bool enabled; }
    public class MonoBehaviour : Behaviour {
        public void Invoke(string m, float t) {} public void CancelInvoke() {}
        public void StartCoroutine(System.Collections.IEnumerator e) {}
    }
    public class GameObject : Object {
        public Transform transform; public bool activeInHierarchy; public bool activeSelf; public string tag;
        public void SetActive(bool b) {}
        public T GetComponent<T>() { return default(T); }
        public T AddComponent<T>() where T : Component { return default(T); }
        public T GetComponentInChildren<T>() { return default(T); }
        public static GameObject FindWithTag(string t) { return null; }
        public static GameObject[] FindGameObjectsWithTag(string t) { return new GameObject[0]; }
        public static GameObject Find(string n) { return null; }
        public bool CompareTag(string t) { return false; }
    }
    public class Transform : Component, System.Collections.IEnumerable {
        public Vector3 position, localPosition, forward, right, up, eulerAngles, localEulerAngles, localScale;
        public Quaternion rotation, localRotation;
        public Transform parent;
        public int childCount;
        public void LookAt(Vector3 v) {} public void LookAt(Transform t) {}
        public void Rotate(Vector3 v) {} public void Rotate(float x, float y, float z) {} public void Rotate(Vector3 axis, float a) {}
        public void Translate(Vector3 v) {} public void Translate(Vector3 v, Space s) {}
        public Transform GetChild(int i) { return null; }
        public Vector3 TransformDirection(Vector3 v) { return v; }
        public System.Collections.IEnumerator GetEnumerator() { yield break; }
    }
    public enum Space { World, Self }
    public struct Vector3 {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public Vector3(float x, float y) { this.x = x; this.y = y; this.z = 0; }
        public static Vector3 zero, one, up, down, forward, back, right, left;
        public float magnitude { get { return 0; } } public float sqrMagnitude { get { return 0; } }
        public Vector3 normalized { get { return this; } }
        public static float Distance(Vector3 a, Vector3 b) { return 0; }
        public static float Dot(Vector3 a, Vector3 b) { return 0; }
        public static float Angle(Vector3 a, Vector3 b) { return 0; }
        public static Vector3 Lerp(Vector3 a, Vector3 b, float t) { return a; }
        public static Vector3 MoveTowards(Vector3 a, Vector3 b, float t) { return a; }
        public static Vector3 ClampMagnitude(Vector3 a, float m) { return a; }
        public static Vector3 operator +(Vector3 a, Vector3 b) { return a; }
        public static Vector3 operator -(Vector3 a, Vector3 b) { return a; }
        public static Vector3 operator -(Vector3 a) { return a; }
        public static Vector3 operator *(Vector3 a, float b) { return a; }
        public static Vector3 operator *(float b, Vector3 a) { return a; }
        public static Vector3 operator /(Vector3 a, float b) { return a; }
        public static bool operator ==(Vector3 a, Vector3 b) { return true; }
        public static bool operator !=(Vector3 a, Vector3 b) { return false; }
        public override bool Equals(object o) { return true; } public override int GetHashCode() { return 0; }
    }
    public struct Vector2 { public float x, y; public Vector2(float x, float y) { this.x = x; this.y = y; } }
    public struct Quaternion {
        public static Quaternion identity;
        public Vector3 eulerAngles;
        public static Quaternion Euler(float x, float y, float z) { return identity; }
        public static Quaternion Euler(Vector3 v) { return identity; }
        public static Quaternion LookRotation(Vector3 v) { return identity; }
        public static Quaternion Slerp(Quaternion a, Quaternion b, float t) { return a; }
        public static Quaternion Lerp(Quaternion a, Quaternion b, float t) { return a; }
        public static Quaternion AngleAxis(float a, Vector3 v) { return identity; }
        public static Vector3 operator *(Quaternion q, Vector3 v) { return v; }
        public static Quaternion operator *(Quaternion a, Quaternion b) { return a; }
    }
    public struct Color {
        public float r, g, b, a;
        public Color(float r, float g, float b, float a) { this.r = r; this.g = g; this.b = b; this.a = a; }
        public Color(float r, float g, float b) { this.r = r; this.g = g; this.b = b; this.a = 1; }
        public static Color red, green, blue, yellow, white, black, gray, cyan, magenta, clear;
    }
    public static class Mathf {
        public const float PI = 3.14159f, Deg2Rad = 0.01745f, Rad2Deg = 57.29f, Infinity = float.PositiveInfinity;
        public static float Clamp(float v, float a, float b) { return v; } public static int Clamp(int v, int a, int b) { return v; }
        public static float Clamp01(float v) { return v; }
        public static float Max(float a, float b) { return a; } public static int Max(int a, int b) { return a; } public static float Max(params float[] a) { return 0; }
        public static float Min(float a, float b) { return a; } public static int Min(int a, int b) { return a; }
        public static float Pow(float a, float b) { return a; } public static float Exp(float a) { return a; }
        public static float Abs(float a) { return a; } public static int Abs(int a) { return a; }
        public static float Sin(float a) { return a; } public static float Cos(float a) { return a; } public static float Sqrt(float a) { return a; }
        public static float Lerp(float a, float b, float t) { return a; } public static float Log(float a) { return a; }
        public static int RoundToInt(float a) { return 0; } public static int FloorToInt(float a) { return 0; } public static int CeilToInt(float a) { return 0; }
        public static float Sign(float a) { return a; } public static float Atan2(float y, float x) { return 0; }
        public static float MoveTowards(float a, float b, float d) { return a; }
        public static bool Approximately(float a, float b) { return true; }
    }
    public static class Random {
        public static float value; public static Vector3 insideUnitSphere; public static Vector2 insideUnitCircle;
        public static float Range(float a, float b) { return a; } public static int Range(int a, int b) { return a; }
    }
    public static class Debug {
        public static void Log(object o) {} public static void LogWarning(object o) {} public static void LogError(object o) {}
        public static void DrawLine(Vector3 a, Vector3 b, Color c) {} public static void DrawRay(Vector3 a, Vector3 b, Color c) {}
    }
    public static class Time { public static float deltaTime, unscaledDeltaTime, time, fixedDeltaTime, timeScale; }
    public enum KeyCode { None, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Alpha0, Alpha1, Alpha2, Alpha3, Alpha4, Alpha5, Alpha6, Alpha7, Alpha8, Alpha9,
        Escape, Space, LeftShift, RightShift, LeftControl, Tab, Return, Mouse0, Mouse1 }
    public static class Input {
        public static Vector3 mousePosition; public static Vector2 mouseScrollDelta;
        public static bool GetKey(KeyCode k) { return false; } public static bool GetKeyDown(KeyCode k) { return false; } public static bool GetKeyUp(KeyCode k) { return false; }
        public static float GetAxis(string a) { return 0; } public static float GetAxisRaw(string a) { return 0; }
        public static bool GetMouseButton(int b) { return false; } public static bool GetMouseButtonDown(int b) { return false; }
    }
    public enum CursorLockMode { None, Locked, Confined }
    public static class Cursor { public static CursorLockMode lockState; public static bool visible; }
    public class Camera : Behaviour { public static Camera main; public Ray ScreenPointToRay(Vector3 p) { return new Ray(); } public float fieldOfView; }
    public struct Ray { public Vector3 origin, direction; public Ray(Vector3 o, Vector3 d) { origin = o; direction = d; } }
    public struct RaycastHit { public Collider collider; public Vector3 point; public float distance; public Transform transform; }
    public class Collider : Component { }
    public class CharacterController : Collider { public bool isGrounded; public void Move(Vector3 v) {} public bool enabled; }
    public class Rigidbody : Component { public Vector3 velocity; public bool isKinematic; }
    public class Renderer : Component { public bool enabled; }
    public struct LayerMask { public int value; public static implicit operator int(LayerMask m) { return m.value; } public static implicit operator LayerMask(int v) { var m = new LayerMask(); m.value = v; return m; } public static int GetMask(params string[] n) { return 0; } }
    public static class Physics {
        public static bool Raycast(Vector3 o, Vector3 d, out RaycastHit h, float m, int mask) { h = new RaycastHit(); return false; }
        public static bool Raycast(Vector3 o, Vector3 d, float m, int mask) { return false; }
        public static bool Linecast(Vector3 a, Vector3 b, int mask) { return false; }
        public static bool Linecast(Vector3 a, Vector3 b, out RaycastHit h, int mask) { h = new RaycastHit(); return false; }
        public static Collider[] OverlapSphere(Vector3 c, float r) { return new Collider[0]; }
        public static bool Raycast(Ray r, out RaycastHit h, float d) { h = new RaycastHit(); return false; }
        public static bool Raycast(Vector3 o, Vector3 d, out RaycastHit h, float m) { h = new RaycastHit(); return false; }
        public static bool Raycast(Vector3 o, Vector3 d, float m) { return false; }
        public static bool Linecast(Vector3 a, Vector3 b) { return false; }
    }
    public static class Gizmos { public static Color color; public static void DrawWireSphere(Vector3 c, float r) {} public static void DrawLine(Vector3 a, Vector3 b) {} public static void DrawSphere(Vector3 c, float r) {} }
    [AttributeUsage(AttributeTargets.All, AllowMultiple = true)] public class HeaderAttribute : Attribute { public HeaderAttribute(string h) {} }
    [AttributeUsage(AttributeTargets.All)] public class TooltipAttribute : Attribute { public TooltipAttribute(string h) {} }
    [AttributeUsage(AttributeTargets.All)] public class SerializeField : Attribute { }
    [AttributeUsage(AttributeTargets.All)] public class RangeAttribute : Attribute { public RangeAttribute(float a, float b) {} }
    [AttributeUsage(AttributeTargets.All)] public class TextAreaAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.All)] public class SpaceAttribute : Attribute { public SpaceAttribute() {} public SpaceAttribute(float h) {} }
    [AttributeUsage(AttributeTargets.All)] public class RequireComponent : Attribute { public RequireComponent(Type t) {} }
}

namespace UnityEngine.AI
{
    public class NavMeshAgent : UnityEngine.Behaviour {
        public float speed, stoppingDistance, remainingDistance, angularSpeed, acceleration;
        public bool isStopped, pathPending, hasPath, isOnNavMesh;
        public UnityEngine.Vector3 velocity, destination;
        public bool SetDestination(UnityEngine.Vector3 v) { return true; }
        public void ResetPath() {} public bool Warp(UnityEngine.Vector3 v) { return true; }
    }
    public struct NavMeshHit { public UnityEngine.Vector3 position; public bool hit; }
    public static class NavMesh {
        public const int AllAreas = -1;
        public static bool SamplePosition(UnityEngine.Vector3 p, out NavMeshHit h, float d, int mask) { h = new NavMeshHit(); return true; }
    }
}

namespace Unity.AI.Navigation
{
    public class NavMeshSurface : UnityEngine.MonoBehaviour { public void BuildNavMesh() {} }
}

namespace Unity.MLAgents
{
    using Unity.MLAgents.Sensors;
    using Unity.MLAgents.Actuators;
    public enum StatAggregationMethod { Average, MostRecent, Sum, Histogram }
    public class StatsRecorder { public void Add(string k, float v, StatAggregationMethod m = StatAggregationMethod.Average) {} }
    public class EnvironmentParameters { public float GetWithDefault(string k, float d) { return d; } }
    public class Academy {
        public static Academy Instance; public static bool IsInitialized;
        public StatsRecorder StatsRecorder; public EnvironmentParameters EnvironmentParameters;
        public bool IsCommunicatorOn; public int StepCount; public int TotalStepCount;
    }
    public class Agent : UnityEngine.MonoBehaviour {
        public int MaxStep; public int StepCount; public int CompletedEpisodes;
        public virtual void Initialize() {} public virtual void OnEpisodeBegin() {}
        public virtual void CollectObservations(VectorSensor s) {}
        public virtual void OnActionReceived(ActionBuffers a) {}
        public virtual void Heuristic(in ActionBuffers a) {}
        public virtual void WriteDiscreteActionMask(IDiscreteActionMask m) {}
        public void AddReward(float r) {} public void SetReward(float r) {} public float GetCumulativeReward() { return 0; }
        public void EndEpisode() {} public void EpisodeInterrupted() {} public void RequestDecision() {} public void RequestAction() {}
        protected virtual void OnEnable() {} protected virtual void OnDisable() {}
    }
}
namespace Unity.MLAgents.Sensors
{
    public class VectorSensor {
        public void AddObservation(float v) {} public void AddObservation(int v) {} public void AddObservation(bool v) {}
        public void AddObservation(UnityEngine.Vector3 v) {} public void AddObservation(UnityEngine.Vector2 v) {}
        public void AddObservation(UnityEngine.Quaternion v) {}
        public void AddObservation(System.Collections.Generic.IList<float> v) {}
        public void AddOneHotObservation(int i, int n) {}
    }
}
namespace Unity.MLAgents.Actuators
{
    public struct ActionSegment<T> where T : struct {
        public T this[int i] { get { return default(T); } set {} }
        public int Length;
    }
    public struct ActionBuffers { public ActionSegment<float> ContinuousActions; public ActionSegment<int> DiscreteActions; }
    public interface IDiscreteActionMask { void SetActionEnabled(int branch, int actionIndex, bool isEnabled); }
}
