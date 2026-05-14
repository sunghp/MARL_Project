using UnityEngine;
using System.Collections.Generic;

public class RoleManager : MonoBehaviour
{
    // ===== 싱글톤 패턴 =====
    public static RoleManager Instance { get; private set; }

    // ===== 역할 종류 =====
    public enum Role
    {
        Human,      // 일반 인간
        Saboteur,   // 사보타주
        Captain     // 함장 (인간 팀)
    }

    // ===== 캐릭터-역할 매핑 =====
    private Dictionary<GameObject, Role> characterRoles = new Dictionary<GameObject, Role>();

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

    // ===== 역할 배정 (게임 시작 시 호출) =====
    public void AssignRoles(List<GameObject> allCharacters)
    {
        if (allCharacters == null || allCharacters.Count < 3)
        {
            Debug.LogError("최소 3명 이상의 캐릭터가 필요합니다!");
            return;
        }

        // 리스트 복사 후 셔플
        List<GameObject> shuffled = new List<GameObject>(allCharacters);
        ShuffleList(shuffled);

        int sabotageCount = GameManager.Instance.sabotageCount;
        int totalCount = shuffled.Count;

        Debug.Log("========== 역할 배정 시작 ==========");

        // 1. 먼저 사보타주 배정
        for (int i = 0; i < sabotageCount && i < totalCount; i++)
        {
            AssignRole(shuffled[i], Role.Saboteur);
        }

        // 2. 나머지 중에서 함장 1명 선정 (인간 중 랜덤)
        List<GameObject> remainingHumans = shuffled.GetRange(sabotageCount, totalCount - sabotageCount);
        
        if (remainingHumans.Count > 0)
        {
            // 랜덤으로 함장 선정
            int captainIndex = Random.Range(0, remainingHumans.Count);
            AssignRole(remainingHumans[captainIndex], Role.Captain);
            remainingHumans.RemoveAt(captainIndex);
        }

        // 3. 나머지는 일반 인간
        foreach (GameObject human in remainingHumans)
        {
            AssignRole(human, Role.Human);
        }

        Debug.Log("========== 역할 배정 완료 ==========");
        PrintRoleSummary();
    }

    // ===== 개별 역할 배정 =====
    void AssignRole(GameObject character, Role role)
    {
        characterRoles[character] = role;

        if (role == Role.Captain)
        {
            if (character.GetComponent<CaptainGun>() == null)
            {
                character.AddComponent<CaptainGun>();
            }
        }

        bool isSaboteur = (role == Role.Saboteur);
        bool isCaptain = (role == Role.Captain);

        // GameManager에 등록
        GameManager.Instance.RegisterCharacter(character, isSaboteur, isCaptain);
    }

    // ===== 리스트 셔플 (Fisher-Yates) =====
    void ShuffleList(List<GameObject> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int randomIndex = Random.Range(0, i + 1);
            GameObject temp = list[i];
            list[i] = list[randomIndex];
            list[randomIndex] = temp;
        }
    }

    // ===== 역할 확인 메서드 =====
    public Role GetRole(GameObject character)
    {
        if (characterRoles.TryGetValue(character, out Role role))
        {
            return role;
        }
        Debug.LogWarning($"{character.name}의 역할이 배정되지 않았습니다!");
        return Role.Human; // 기본값
    }

    public bool IsSaboteur(GameObject character)
    {
        return GetRole(character) == Role.Saboteur;
    }

    public bool IsCaptain(GameObject character)
    {
        return GetRole(character) == Role.Captain;
    }

    public bool IsHumanTeam(GameObject character)
    {
        Role role = GetRole(character);
        return role == Role.Human || role == Role.Captain;
    }

    // ===== 역할 정보 출력 (디버그용) =====
    void PrintRoleSummary()
    {
        Debug.Log($"인간 팀: {GameManager.Instance.GetAliveHumanCount()}명");
        Debug.Log($"사보타주: {GameManager.Instance.GetAliveSaboteurCount()}명");
    }

    // ===== 함장 공개 (게임 시작 시 모두에게 알림) =====
    public void AnnounceCaptain()
    {
        if (GameManager.Instance.captain != null)
        {
            string captainName = GameManager.Instance.captain.name;
            Debug.Log($"📢 [전체 공지] 이번 항해의 함장은 {captainName}입니다!");
        }
    }

    // ===== 특정 캐릭터에게만 역할 공개 (자기 역할 확인) =====
    public string GetRoleDescription(GameObject character)
    {
        Role role = GetRole(character);

        return role switch
        {
            Role.Captain => "당신은 함장입니다. 총 2발로 사보타주를 찾아 처형하세요.",
            Role.Saboteur => "당신은 사보타주입니다. 우주선을 파괴하거나 인간과 동수가 되면 승리합니다.",
            Role.Human => "당신은 선원입니다. 우주선을 수리하고 목적지까지 생존하세요.",
            _ => "역할 없음"
        };
    }

    // ===== 사보타주끼리 서로 확인 =====
    public List<GameObject> GetSaboteurList()
    {
        return new List<GameObject>(GameManager.Instance.saboteurs);
    }

    public string GetSaboteurNames()
    {
        List<GameObject> saboteurs = GameManager.Instance.saboteurs;
        if (saboteurs.Count == 0) return "없음";

        List<string> names = new List<string>();
        foreach (var sab in saboteurs)
        {
            names.Add(sab.name);
        }
        return string.Join(", ", names);
    }

    // ===== 역할 리셋 =====

    public void ResetRoles()
    {
        characterRoles.Clear();  // ✅ 이게 RoleManager의 실제 데이터
        GameManager.Instance.saboteurs.Clear();
        GameManager.Instance.humans.Clear();
        GameManager.Instance.captain = null;
        
        Debug.Log("========== 역할 초기화 ==========");
    }
}
