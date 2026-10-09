# C# 컴파일 검사 (Unity 없이)

`Assets/Scripts/*.cs`를 Unity 없이 컴파일해 문법 오류와 스크립트 간 참조 오류(없는 메서드/필드/타입)를 잡는다.

```bash
cd tools/csharp_check
dotnet build -nologo -v q      # .NET 8 SDK 필요
```

- `UnityStubs.cs`는 스크립트가 쓰는 Unity / ML-Agents API의 **시그니처만** 흉내낸 가짜 구현이다.
- 통과해도 Unity 고유 동작(직렬화, NavMesh, ML-Agents 통신)은 보장하지 않는다. 최종 확인은 Unity 에디터에서 한다.
- 스크립트에 새 Unity API를 쓰면 스텁에 추가해야 할 수 있다.
