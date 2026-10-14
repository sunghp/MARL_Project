"""
plot_saboteur_vs_bot.py - 사보타주 vs 규칙봇 인간팀 학습 곡선

  python sim/plot_saboteur_vs_bot.py --run A=dirA_s0,dirA_s1 --run B=dirB_s0,dirB_s1 \
      --ref ladder.json --out sim/results/saboteur_vs_bot

각 실행의 games.jsonl(학습 게임 = 규칙봇 인간팀 상대)과 TensorBoard 엔트로피를 읽는다.
--ref: {"규칙봇 사보타주": 0.375, ...} 형태의 기준 승률 (가로선)
"""
import argparse
import json
import os

import numpy as np

from analyze import COLORS, bot_train_games, load_games, load_tb, plt, rolling, train_step_axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="이름=dir1,dir2 (이름 뒤 :설명 가능)")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    refs = json.load(open(args.ref, encoding="utf-8")) if args.ref else {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_win, ax_sab), (ax_dead, ax_ent) = axes
    summary = {}
    for spec in args.run:
        name, dirs = spec.split("=", 1)
        label = name
        if ":" in name:
            name, label = name.split(":", 1)
        color = COLORS.get(name[0], None)
        for i, d in enumerate(dirs.split(",")):
            games = load_games(d)
            tg = bot_train_games(games)
            x = train_step_axis(games, bot=True) / 1e6
            n = min(len(x), len(tg))
            x, tg = x[:n], tg[:n]
            lab = f"{label} (시드 {i})"
            ls = "-" if i == 0 else "--"
            w = args.window
            ax_win.plot(x, rolling([1 - g["human_win"] for g in tg], w), ls, color=color, label=lab)
            ax_sab.plot(x, rolling([g["sab_total"] for g in tg], w), ls, color=color, label=f"{lab} 부수기")
            ax_sab.plot(x, rolling([g["sab_total"] - g["sab_hidden"] for g in tg], w), ":", color=color, alpha=0.8,
                        label=f"{lab} 들킨 부수기")
            ax_dead.plot(x, rolling([g["shots_hit"] for g in tg], w), ls, color=color, label=f"{lab} 처형된 사보타주")
            tb = load_tb(d)
            ent = tb.get("policy/entropy_saboteur", [])
            if ent:
                ent.sort()
                ax_ent.plot([s / 1e6 for s, _ in ent], rolling([v for _, v in ent], 50), ls, color=color, label=lab)
            last = tg[-500:]
            summary[f"{name}_s{i}"] = {
                "games": len(tg),
                "steps": float(x[-1]) if len(x) else 0.0,
                "sab_win_last500": round(1 - float(np.mean([g["human_win"] for g in last])), 3),
                "sab_last500": round(float(np.mean([g["sab_total"] for g in last])), 2),
                "seen_sab_last500": round(float(np.mean([g["sab_total"] - g["sab_hidden"] for g in last])), 2),
                "executed_last500": round(float(np.mean([g["shots_hit"] for g in last])), 2),
                "destroy_last500": round(float(np.mean(["파괴" in g["reason"] for g in last])), 3),
                "entropy_last": round(float(np.mean([v for _, v in ent[-50:]])), 2) if ent else None,
            }
    prev = None
    for k, v in sorted(refs.items(), key=lambda kv: -kv[1]):
        ax_win.axhline(v, color="gray", lw=1, ls="--")
        y = v + 0.01 if prev is None or prev - v > 0.05 else v - 0.04     # 가까운 선은 글자를 아래로
        ax_win.text(0.99, y, f"{k} {v:.0%}", color="gray", fontsize=9, ha="right",
                    transform=ax_win.get_yaxis_transform())
        prev = v
    ax_win.set_title(f"사보타주 승률 (규칙봇 인간팀 상대, 최근 {args.window}판)")
    ax_win.set_ylim(0, 1)
    ax_sab.set_title("게임당 부수기 (실선 전체, 점선 들킨 것)")
    ax_dead.set_title("게임당 처형된 사보타주 (최대 2)")
    ax_ent.set_title("사보타주 정책 엔트로피 (균등분포 4.39)")
    for ax in axes.flat:
        ax.set_xlabel("학습 스텝 (M)")
        ax.grid(alpha=0.3)
    ax_win.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax_ent.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curves.png"), dpi=110)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
