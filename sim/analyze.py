"""
analyze.py - 가상 환경 학습 결과 시각화

  python sim/analyze.py --run A=dirA_s0,dirA_s1 --run B=dirB_s0,dirB_s1 --out sim/results

각 실행 디렉터리의 games.jsonl(게임별 ground truth)과 runs/mappo(TensorBoard)를 읽어
학습 곡선 PNG와 summary.json을 만든다.
"""
import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for f in sorted(glob.glob("/usr/share/fonts/**/NanumGothic.ttf", recursive=True)) + glob.glob("/usr/share/fonts/**/*CJK*.tt*", recursive=True):
    font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    break
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"A": "#d1495b", "B": "#2e86ab", "C": "#3b8b3b", "D": "#8f5fbf", "E": "#e08a00", "F": "#1f77b4", "G": "#2ca02c", "H": "#d62728", "I": "#9467bd", "J": "#e377c2", "K": "#17becf", "L": "#bcbd22"}
REASON_GROUPS = [
    ("목적지 도착", lambda r: r == "목적지 도착"),
    ("사보타주 전원 처형", lambda r: r == "모든 사보타주 제거"),
    ("인간과 동수(오사)", lambda r: r == "인간과 동수"),
    ("방 완전 파괴", lambda r: "완전 파괴" in r),
    ("평균 안정도 붕괴", lambda r: "평균 안정도" in r),
    ("시간 초과", lambda r: "시간 초과" in r),
]


def load_games(d):
    with open(os.path.join(d, "games.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_tb(d):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for ev in glob.glob(os.path.join(d, "runs", "mappo", "events.*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            out.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    return out


def rolling(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    r = (c[w:] - c[:-w]) / w
    head = [x[: i + 1].mean() for i in range(w - 1)]
    return np.concatenate([head, r])


def train_games(games):
    """학습 게임 중 학습 정책끼리 붙은 게임 (평가, 커리큘럼 봇 게임 제외)"""
    return [g for g in games if not g["eval_mode"] and not g.get("bot_team", 0)]


def bot_train_games(games):
    """커리큘럼 봇 게임 (학습 중 한 팀이 규칙봇)"""
    return [g for g in games if not g["eval_mode"] and g.get("bot_team", 0)]


def eval_blocks(games):
    """연속된 평가 게임을 블록으로 묶어 (스텝, 사보타주 승률, 인간 승률) 반환"""
    blocks, cur = [], []
    for g in games:
        if g["eval_mode"]:
            cur.append(g)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    out = []
    for b in blocks:
        sab = [1 - g["human_win"] for g in b if g["frozen_team"] == 2]
        hum = [g["human_win"] for g in b if g["frozen_team"] == 1]
        out.append((b[0]["agent_decisions"],
                    float(np.mean(sab)) if sab else np.nan,
                    float(np.mean(hum)) if hum else np.nan, len(sab), len(hum)))
    return out


def train_step_axis(games, bot=False):
    """평가에 쓴 decision을 빼고 '학습 스텝' 축으로 변환 (bot=True면 봇 게임 위치, 아니면 정책끼리 게임 위치)"""
    xs, offset, prev = [], 0, 0
    for g in games:
        if g["eval_mode"]:
            offset += g["agent_decisions"] - prev
        prev = g["agent_decisions"]
        if not g["eval_mode"] and bool(g.get("bot_team", 0)) == bot:
            xs.append(g["agent_decisions"] - offset)
    return np.array(xs)


def summarize_run(games, tb):
    tg = train_games(games)
    n = len(tg)
    q = max(1, n // 5)
    first, last = tg[:q], tg[-q:]

    def stats(gs):
        reasons = collections.Counter()
        for g in gs:
            for name, fn in REASON_GROUPS:
                if fn(g["reason"]):
                    reasons[name] += 1
        return {
            "games": len(gs),
            "human_win_rate": round(float(np.mean([g["human_win"] for g in gs])), 3),
            "duration_s": round(float(np.mean([g["duration"] for g in gs])), 1),
            "sabotage_per_game": round(float(np.mean([g["sab_total"] for g in gs])), 2),
            "hidden_rate": round(float(np.sum([g["sab_hidden"] for g in gs]) / max(1, np.sum([g["sab_total"] for g in gs]))), 3),
            "repairs_per_game": round(float(np.mean([g["repairs"] for g in gs])), 2),
            "interrupts_per_game": round(float(np.mean([g.get("sab_interrupted", 0) for g in gs])), 2),
            "shots_per_game": round(float(np.mean([g["shots"] for g in gs])), 2),
            "shot_accuracy": round(float(np.sum([g["shots_hit"] for g in gs]) / max(1, np.sum([g["shots"] for g in gs]))), 3),
            "witness_per_game": round(float(np.mean([g.get("witness_events", 0) for g in gs])), 2),
            "shot_in_first_5s": round(float(np.mean([g["duration"] <= 5.0 and g["shots"] > 0 for g in gs])), 3),
            "reasons": {k: round(v / len(gs), 3) for k, v in reasons.most_common()},
        }

    ent = {}
    for tag in ("policy/entropy_human", "policy/entropy_saboteur", "policy/entropy_captain"):
        v = [y for _, y in sorted(tb.get(tag, []))]
        if v:
            ent[tag.split("_")[-1]] = {"start": round(v[0], 3), "end": round(float(np.mean(v[-20:])), 3)}
    bg = bot_train_games(games)
    return {
        "train_games": n,
        "bot_train_games": len(bg),
        "bot_games_human_win_last20pct": (round(float(np.mean([g["human_win"] for g in bg[-max(1, len(bg) // 5):]])), 3)
                                          if bg else None),
        "trainer_counted_episodes": int(max([s for s, _ in tb.get("episode/total_reward", [(0, 0)])])),
        "first_20pct": stats(first),
        "last_20pct": stats(last),
        "eval_blocks": [
            {"at_decisions": b[0], "saboteur_vs_bot": round(b[1], 3), "human_vs_bot": round(b[2], 3),
             "n_sab": b[3], "n_hum": b[4]} for b in eval_blocks(games)],
        "entropy": ent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="이름=dir1,dir2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None, help="baselines.py 결과 json lines")
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for spec in args.run:
        name, dirs = spec.split("=", 1)
        runs[name] = [(d, load_games(d), load_tb(d)) for d in dirs.split(",")]

    base = {}
    if args.baseline and os.path.exists(args.baseline):
        for l in open(args.baseline, encoding="utf-8"):
            r = json.loads(l)
            base[r["name"]] = r

    W = args.window
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    ax = axes.ravel()

    for name, items in runs.items():
        col = COLORS.get(name[0], None)
        for i, (d, games, tb) in enumerate(items):
            tg = train_games(games)
            if not tg:
                continue
            x = train_step_axis(games) / 1e6
            lab = name if i == 0 else None
            ls = "-" if i == 0 else "--"
            ax[0].plot(x, rolling([g["human_win"] for g in tg], W), color=col, ls=ls, label=lab)
            bg = bot_train_games(games)
            if bg:
                ax[0].plot(train_step_axis(games, bot=True) / 1e6, rolling([g["human_win"] for g in bg], W),
                           color=col, ls=":", alpha=0.6, label=f"{name} (봇 사보타주 상대)" if i == 0 else None)
            ax[1].plot(x, rolling([g["duration"] for g in tg], W), color=col, ls=ls, label=lab)
            ax[2].plot(x, rolling([g["shots"] for g in tg], W), color=col, ls=ls, label=lab)
            ax[3].plot(x, rolling([g["duration"] <= 5.0 and g["shots"] > 0 for g in tg], W), color=col, ls=ls, label=lab)
            ax[4].plot(x, rolling([g["sab_total"] for g in tg], W), color=col, ls=ls, label=lab)
            ax[5].plot(x, rolling([g["repairs"] for g in tg], W), color=col, ls=ls, label=lab)
            for tag, a, sty in (("policy/entropy_captain", ax[6], ":"), ("policy/entropy_saboteur", ax[6], "-."),
                                ("policy/entropy_human", ax[6], "-")):
                v = sorted(tb.get(tag, []))
                if v and i == 0:
                    a.plot([s / 1e6 for s, _ in v], rolling([y for _, y in v], 20), color=col, ls=sty,
                           label=f"{name} {tag.split('_')[-1]}")
            eb = eval_blocks(games)
            if eb:
                # 평가 블록 위치를 학습 스텝 축으로 근사 (평가 시작 시점의 학습 스텝)
                xs_all = train_step_axis(games)
                starts = []
                for b in eb:
                    idx = np.searchsorted([g["agent_decisions"] for g in tg], b[0])
                    starts.append((xs_all[min(idx, len(xs_all) - 1)] if len(xs_all) else 0) / 1e6)
                ax[7].plot(starts, [b[1] for b in eb], color=col, ls=ls, marker="o", label=f"{name} 사보타주 vs 봇" if i == 0 else None)
                ax[7].plot(starts, [b[2] for b in eb], color=col, ls=ls, marker="s", alpha=0.6, label=f"{name} 인간팀 vs 봇" if i == 0 else None)

    # 종료 사유 비율 (마지막 run 그룹별 첫/마지막 20%)
    labels, data = [], []
    for name, items in runs.items():
        for tag, sel in (("초반", lambda tg: tg[: max(1, len(tg) // 5)]), ("후반", lambda tg: tg[-max(1, len(tg) // 5):])):
            gs = [g for _, games, _ in items for g in sel(train_games(games))]
            cnt = collections.Counter()
            for g in gs:
                for rn, fn in REASON_GROUPS:
                    if fn(g["reason"]):
                        cnt[rn] += 1
            labels.append(f"{name}\n{tag}")
            data.append([cnt[rn] / max(1, len(gs)) for rn, _ in REASON_GROUPS])
    data = np.array(data)
    bottom = np.zeros(len(labels))
    cmap = ["#4c9f70", "#2e86ab", "#f18f01", "#c73e1d", "#6c4f77", "#999999"]
    for j, (rn, _) in enumerate(REASON_GROUPS):
        ax[8].bar(labels, data[:, j], bottom=bottom, color=cmap[j], label=rn)
        bottom += data[:, j]

    if "random_vs_random" in base:
        ax[0].axhline(base["random_vs_random"]["human_win_rate"], color="gray", ls=":", label="무작위 정책")
    if "random_sab_vs_bot_humans" in base:
        ax[7].axhline(1 - base["random_sab_vs_bot_humans"]["human_win_rate"], color="gray", ls=":", label="무작위 사보타주 vs 봇")
    if "random_humans_vs_bot_sab" in base:
        ax[7].axhline(base["random_humans_vs_bot_sab"]["human_win_rate"], color="gray", ls="--", label="무작위 인간팀 vs 봇")

    titles = [
        ("인간팀 승률 (학습 게임, 이동평균 %d)" % W, "승률"),
        ("게임 길이 (초)", "초"),
        ("게임당 함장 사격 수", "발"),
        ("시작 5초 안에 사격으로 끝난 게임 비율", "비율"),
        ("게임당 부수기 완료 수", "회"),
        ("게임당 실제 수리 수", "회"),
        ("정책 엔트로피 (역할별)", "엔트로피"),
        ("고정 상대(규칙봇) 평가 승률", "승률"),
        ("게임 종료 사유 (학습 초반/후반 20%)", "비율"),
    ]
    for a, (t, yl) in zip(ax, titles):
        a.set_title(t)
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        if a is not ax[8]:
            a.set_xlabel("학습 스텝 (agent-steps, 백만)")
    ax[8].set_xlabel("")
    for a in ax:
        h, l = a.get_legend_handles_labels()
        if h:
            a.legend(fontsize=8)
    fig.suptitle("The Thing MAPPO — 가상 환경 학습 곡선", fontsize=16)
    fig.tight_layout()
    png = os.path.join(args.out, "learning_curves.png")
    fig.savefig(png, dpi=110)

    summary = {name: [dict(dir=os.path.basename(d.rstrip("/")), **summarize_run(g, tb)) for d, g, tb in items]
               for name, items in runs.items()}
    if base:
        summary["baselines"] = base
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(png)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
